import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from rag_engine import query_rag_system, index_single_file, DOC_FOLDER
from auto_crawler import check_and_sync_college_docs

# Supported file formats
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}

# ----------------- BACKGROUND SCHEDULER -----------------
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Run crawler on boot and schedule every 6 hours
    print("🚀 [Startup] Triggering initial college portal crawler sync...")
    try:
        check_and_sync_college_docs()
    except Exception as e:
        print(f"⚠️ Initial crawler sync skipped: {e}")

    scheduler.add_job(check_and_sync_college_docs, "interval", hours=6)
    scheduler.start()
    yield
    # Shutdown
    scheduler.shutdown()


app = FastAPI(title="SNJB Institutional RAG AI", lifespan=lifespan)

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static UI Files
STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
if os.path.exists(STATIC_FOLDER):
    app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


# ----------------- DATA MODELS -----------------
class QueryRequest(BaseModel):
    query: str


# ----------------- API ENDPOINTS -----------------
@app.get("/")
async def serve_frontend():
    """Serves the main frontend index.html."""
    index_path = os.path.join(STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "running", "message": "SNJB Institutional RAG API is live."})


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """Processes student questions through the RAG pipeline."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    answer = query_rag_system(request.query)
    return {"query": request.query, "answer": answer}


@app.post("/admin/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads and indexes PDFs, Word docs, Slides, CSVs, and Images (OCR)."""
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    local_path = os.path.join(DOC_FOLDER, file.filename)

    try:
        # Save file to local folder
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Index into Qdrant Cloud
        chunk_count = index_single_file(local_path, file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": chunk_count,
            "message": f"Successfully indexed '{file.filename}' across {chunk_count} chunk(s)."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.get("/admin/documents")
async def list_documents():
    """Lists all local documents currently cached."""
    files = []
    if os.path.exists(DOC_FOLDER):
        for f in os.listdir(DOC_FOLDER):
            if not f.endswith(".sha256"):
                files.append(f)
    return {"documents": files}


@app.delete("/admin/documents/{filename}")
async def delete_document(filename: str):
    """Deletes a local cached document and its hash file."""
    file_path = os.path.join(DOC_FOLDER, filename)
    hash_path = file_path + ".sha256"

    deleted = False
    if os.path.exists(file_path):
        os.remove(file_path)
        deleted = True
    if os.path.exists(hash_path):
        os.remove(hash_path)

    if deleted:
        return {"status": "success", "message": f"Deleted '{filename}' locally."}
    raise HTTPException(status_code=404, detail="File not found.")