import os
import json
import sqlite3
from datetime import datetime
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

DOC_FOLDER = "./documents"
META_DB_FILE = "doc_metadata.db"
COLLECTION_NAME = "institutional_docs"


# ----------------- SQLITE DATABASE SETUP -----------------
def init_db():
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()

    # 1. Document Metadata Table
    c.execute('''CREATE TABLE IF NOT EXISTS documents 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  filename TEXT UNIQUE, 
                  upload_date TEXT, 
                  file_size_kb REAL, 
                  chunk_count INTEGER)''')

    # 2. Knowledge Gap Logging Table (Admin Analytics)
    c.execute('''CREATE TABLE IF NOT EXISTS knowledge_gaps 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  question TEXT, 
                  timestamp TEXT,
                  status TEXT DEFAULT 'Unresolved')''')

    # 3. Persistent Conversation History Table
    c.execute('''CREATE TABLE IF NOT EXISTS chat_history 
                 (id INTEGER PRIMARY KEY AUTOINCREMENT, 
                  session_id TEXT, 
                  role TEXT, 
                  content TEXT, 
                  timestamp TEXT)''')

    conn.commit()
    conn.close()


init_db()


# ----------------- DOCUMENT MANAGEMENT -----------------
def register_document(filename, file_size_kb, chunk_count):
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute('''INSERT OR REPLACE INTO documents (filename, upload_date, file_size_kb, chunk_count)
                 VALUES (?, ?, ?, ?)''',
              (filename, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file_size_kb, chunk_count))
    conn.commit()
    conn.close()


def get_all_documents():
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT filename, upload_date, file_size_kb, chunk_count FROM documents ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"filename": r[0], "upload_date": r[1], "size_kb": r[2], "chunks": r[3]} for r in rows]


def delete_document(filename):
    file_path = os.path.join(DOC_FOLDER, filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("DELETE FROM documents WHERE filename = ?", (filename,))
    conn.commit()
    conn.close()


# ----------------- KNOWLEDGE GAP ANALYTICS -----------------
def log_unanswered_gap(query):
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO knowledge_gaps (question, timestamp) VALUES (?, ?)",
              (query, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_knowledge_gaps():
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT id, question, timestamp, status FROM knowledge_gaps ORDER BY id DESC")
    rows = c.fetchall()
    conn.close()
    return [{"id": r[0], "question": r[1], "timestamp": r[2], "status": r[3]} for r in rows]


def resolve_gap(gap_id):
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("UPDATE knowledge_gaps SET status = 'Resolved' WHERE id = ?", (gap_id,))
    conn.commit()
    conn.close()


# ----------------- CONVERSATION MEMORY -----------------
def save_chat_message(session_id, role, content):
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("INSERT INTO chat_history (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
              (session_id, role, content, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit()
    conn.close()


def get_chat_history(session_id, limit=4):
    """Retrieves recent turns for LLM prompt context."""
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content FROM chat_history WHERE session_id = ? ORDER BY id DESC LIMIT ?",
              (session_id, limit))
    rows = c.fetchall()
    conn.close()

    formatted_history = []
    for role, content in reversed(rows):
        formatted_history.append(f"{role.capitalize()}: {content}")
    return "\n".join(formatted_history)


def get_session_history_list(session_id):
    """Retrieves full conversation transcript for frontend restoration."""
    conn = sqlite3.connect(META_DB_FILE)
    c = conn.cursor()
    c.execute("SELECT role, content, timestamp FROM chat_history WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [{"role": r[0], "content": r[1], "timestamp": r[2]} for r in rows]


# ----------------- OPTIMIZED QDRANT & EMBEDDING CACHING -----------------
@lru_cache(maxsize=1)
def get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    # Model loaded once in memory for all future queries
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")


@lru_cache(maxsize=1)
def get_vector_db():
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient

    q_url = os.getenv("QDRANT_URL")
    q_key = os.getenv("QDRANT_API_KEY")

    if not q_url or not q_key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY is missing from environment variables.")

    # Reuses HTTP connection pool
    client = QdrantClient(url=q_url, api_key=q_key, timeout=60)

    return QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=get_embeddings()
    )


def index_single_pdf(file_path, fname):
    """Fast incremental upload with batch protection."""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_qdrant import QdrantVectorStore

    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)

    loader = PyPDFLoader(file_path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["source_file"] = fname

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    file_size = round(os.path.getsize(file_path) / 1024, 2)
    register_document(fname, file_size, len(chunks))

    if chunks:
        q_url = os.getenv("QDRANT_URL")
        q_key = os.getenv("QDRANT_API_KEY")

        QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=get_embeddings(),
            url=q_url,
            api_key=q_key,
            collection_name=COLLECTION_NAME,
            force_recreate=False,
            batch_size=64
        )
    return len(chunks)


# ----------------- MAIN QUERY ENGINE -----------------
def answer_query(query, session_id="default_user"):
    from langchain_groq import ChatGroq

    # 1. Retrieve recent history context
    history_str = get_chat_history(session_id)

    # 2. Fast Small-Talk / Greeting Filter
    clean_q = query.strip().lower()
    greetings = ["hello", "hi", "hey", "good morning", "good evening", "who are you", "start", "menu", "help"]

    if clean_q in greetings:
        res = (
            "Hello! I am InstiAssistant, your official AI academic partner. 👋\n\n"
            "Ask me anything regarding course syllabi, examination guidelines, grading schemes, or institutional policies!"
        )
        save_chat_message(session_id, "user", query)
        save_chat_message(session_id, "assistant", res)
        return res, []

    # 3. Vector DB Search (Optimized k=3 for faster inference)
    try:
        vectordb = get_vector_db()
        results = vectordb.similarity_search_with_score(query, k=3)
    except Exception as e:
        print(f"Vector Search Error: {e}")
        return "I don't know. The vector database is currently unreachable.", []

    if not results:
        log_unanswered_gap(query)
        res = "I don't know. The requested information is not available in the official indexed documents."
        save_chat_message(session_id, "user", query)
        save_chat_message(session_id, "assistant", res)
        return res, []

    context_snippets = []
    sources = []

    for i, (doc, score) in enumerate(results):
        src_file = doc.metadata.get("source_file", "Unknown Document")
        page_num = doc.metadata.get("page", 1)
        snippet_text = doc.page_content.replace("\n", " ").strip()

        context_snippets.append(f"[{i + 1}] File: {src_file} (Page {page_num})\nContent: {doc.page_content}")

        sources.append({
            "file": src_file,
            "page": page_num,
            "snippet": snippet_text[:250] + "..."
        })

    context_str = "\n\n".join(context_snippets)

    # 4. Strictly Grounded Prompt
    prompt = f"""You are an official Institutional AI Assistant.

CRITICAL INSTRUCTIONS:
1. STRICT GROUNDING: Answer using ONLY the official context provided below. Do NOT assume or use outside knowledge.
2. FALLBACK: If the provided context does NOT contain the answer, reply strictly with:
   "I don't know. This information is not available in the indexed documents."
3. STRUCTURED FORMAT:
   - Organize all answers using clear Headings (e.g. ## Overview, ### Course Details).
   - Use Bullet Points (`*`) or numbered lists.
   - Use Bolding (`**...**`) for key terms.
4. CITATIONS: Reference documents inline like [1], [2] next to facts derived from context.

Previous Conversation:
{history_str}

Context Snippets:
{context_str}

Question: {query}
Answer:"""

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke(prompt)
    answer_text = response.content

    if "i don't know" in answer_text.lower():
        log_unanswered_gap(query)

    save_chat_message(session_id, "user", query)
    save_chat_message(session_id, "assistant", answer_text)

    return answer_text, sources