import os
import base64
from io import BytesIO
from PIL import Image
from groq import Groq

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    CSVLoader,
    UnstructuredPowerPointLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq

# ----------------- PATH & ENVIRONMENT SETUP -----------------
DOC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "documents"))
os.makedirs(DOC_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ----------------- LAZY INITIALIZATION (OOM / EXIT 137 PREVENTION) -----------------
_embeddings = None
_qdrant_client = None
_vector_store = None
_llm = None
_groq_native_client = None


def get_embeddings():
    """Lazily loads FastEmbed embeddings (~120MB RAM vs PyTorch's 450MB+)."""
    global _embeddings
    if _embeddings is None:
        from langchain_community.embeddings import FastEmbedEmbeddings
        _embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    return _embeddings


def get_qdrant_client():
    """Lazily initializes the Qdrant Cloud client."""
    global _qdrant_client
    if _qdrant_client is None:
        _qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            timeout=60.0
        )
    return _qdrant_client


def get_vector_store():
    """Lazily connects to the Qdrant Cloud vector store."""
    global _vector_store
    if _vector_store is None:
        _vector_store = QdrantVectorStore(
            client=get_qdrant_client(),
            collection_name="institutional_docs",  # Matches active Qdrant Cloud collection
            embedding=get_embeddings()
        )
    return _vector_store


def get_llm():
    """Lazily initializes the Groq LLM instance."""
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.2
        )
    return _llm


def get_groq_native_client():
    """Lazily initializes the native Groq SDK client for Vision API."""
    global _groq_native_client
    if _groq_native_client is None:
        _groq_native_client = Groq(api_key=GROQ_API_KEY)
    return _groq_native_client


# ----------------- MULTI-FORMAT LOADERS & OCR -----------------
def extract_text_from_image(image_path: str) -> str:
    """Uses Groq Llama 3.2 Vision model to transcribe text, tables, and notices from images."""
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGB":
                img = img.convert("RGB")

            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        client = get_groq_native_client()
        response = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Extract all text, notices, dates, schedules, tables, and academic details from this image. Output clean and structured Markdown text."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"❌ Vision OCR error on {image_path}: {e}")
        return f"[Image transcription failed: {e}]"


def load_document_by_extension(file_path: str):
    """Dynamically routes file parsing based on format extension."""
    ext = os.path.splitext(file_path)[1].lower()

    # Images (Groq Vision OCR)
    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
        text_content = extract_text_from_image(file_path)
        return [Document(page_content=text_content, metadata={"source": os.path.basename(file_path)})]

    # PDFs
    elif ext == ".pdf":
        return PyPDFLoader(file_path).load()

    # Word Documents
    elif ext in [".docx", ".doc"]:
        return Docx2txtLoader(file_path).load()

    # Plain Text & Markdown
    elif ext in [".txt", ".md", ".log"]:
        return TextLoader(file_path, encoding="utf-8").load()

    # CSV Spreadsheets
    elif ext == ".csv":
        return CSVLoader(file_path).load()

    # PowerPoint Slides
    elif ext in [".pptx", ".ppt"]:
        return UnstructuredPowerPointLoader(file_path).load()

    else:
        raise ValueError(f"Unsupported file format: {ext}")


# ----------------- VECTOR INDEXING PIPELINE -----------------
def index_single_file(file_path: str, filename: str) -> int:
    """Loads, chunks, and indexes any supported file or image into Qdrant Cloud."""
    try:
        # 1. Load document content
        raw_docs = load_document_by_extension(file_path)

        # 2. Chunk text intelligently
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(raw_docs)

        # 3. Add explicit metadata tags
        for i, chunk in enumerate(chunks):
            chunk.metadata["source"] = filename
            chunk.metadata["chunk_id"] = i

        # 4. Upsert vectors into Qdrant Cloud via lazy vector store
        vector_store = get_vector_store()
        vector_store.add_documents(chunks)
        print(f"✅ Successfully indexed '{filename}' ({len(chunks)} chunks).")
        return len(chunks)

    except Exception as e:
        print(f"❌ Error indexing '{filename}': {e}")
        raise e


# Backward-compatibility alias for auto_crawler
index_single_pdf = index_single_file


# ----------------- RAG QUERY PROCESSING -----------------
def query_rag_system(user_query: str) -> str:
    """Retrieves top relevant document vectors and generates an answer using Llama-3."""
    try:
        # 1. Similarity Search against Qdrant
        vector_store = get_vector_store()
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        context_docs = retriever.invoke(user_query)

        if not context_docs:
            context_text = "No relevant documents found in the database."
        else:
            context_text = "\n\n---\n\n".join(
                [f"Source: {doc.metadata.get('source', 'Unknown')}\n{doc.page_content}" for doc in context_docs]
            )

        # 2. System Prompt
        system_prompt = f"""You are the official AI Academic Assistant for SNJB College of Engineering.
Answer the student's question accurately based ONLY on the provided document excerpts below.
If the answer cannot be determined from the documents, politely state that the information is not available in the college records.

DOCUMENT CONTEXT:
{context_text}

USER QUESTION:
{user_query}
"""

        # 3. LLM Generation via Groq
        llm = get_llm()
        response = llm.invoke(system_prompt)
        return response.content

    except Exception as e:
        print(f"❌ RAG query error: {e}")
        return "Sorry, I encountered an error while searching the document database. Please try again."