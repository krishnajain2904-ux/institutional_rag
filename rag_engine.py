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
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_groq import ChatGroq

# ----------------- PATH & ENVIRONMENT SETUP -----------------
DOC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "documents"))
os.makedirs(DOC_FOLDER, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# ----------------- INITIALIZE CLIENTS & MODELS -----------------
# 1. Embeddings Model (384 Dimensions)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# 2. Qdrant Cloud Vector Store
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60.0
)

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name="snjb_rag_docs",
    embedding=embeddings
)

# 3. Groq LLM (Text Chat)
llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile",
    temperature=0.2
)

# 4. Groq Native Client (Vision / OCR)
groq_native_client = Groq(api_key=GROQ_API_KEY)


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

        response = groq_native_client.chat.completions.create(
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

    # Images (OCR)
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
    """Loads, chunks, and indexes any supported file/image into Qdrant Cloud."""
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

        # 4. Upsert vectors into Qdrant Cloud
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
        # 1. Similarity Search
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

        # 3. LLM Generation
        response = llm.invoke(system_prompt)
        return response.content

    except Exception as e:
        print(f"❌ RAG query error: {e}")
        return "Sorry, I encountered an error while searching the document database. Please try again."