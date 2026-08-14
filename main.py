import os
import shutil
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

from rag_engine import (
    query_rag_system,
    index_single_file,
    DOC_FOLDER,
    get_qdrant_client
)
from auto_crawler import check_and_sync_college_docs

# Supported file formats (Documents + Images for Groq Vision OCR)
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}

# ----------------- BACKGROUND SCHEDULER & LIFESPAN -----------------
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Manager: Boots the server instantly while running background tasks."""
    print("🚀 [Startup] Launching initial college portal crawler sync in background thread...")

    # 1. Run crawler in a daemon thread so FastAPI binds to $PORT immediately (fixes Render deployment timeouts)
    threading.Thread(target=check_and_sync_college_docs, daemon=True).start()

    # 2. Schedule recurring crawler syncs every 6 hours
    scheduler.add_job(check_and_sync_college_docs, "interval", hours=6)
    scheduler.start()

    yield

    # Shutdown
    scheduler.shutdown()


# Initialize FastAPI Application
app = FastAPI(
    title="SNJB Institutional RAG AI",
    description="Multi-format RAG Assistant & Auto-Crawler for SNJB College of Engineering",
    lifespan=lifespan
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Static Assets Directory
STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
if os.path.exists(STATIC_FOLDER):
    app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


# ----------------- DATA MODELS -----------------
class QueryRequest(BaseModel):
    query: str


# ----------------- FRONTEND & CHAT ENDPOINTS -----------------
@app.get("/")
async def serve_frontend():
    """Serves index.html UI or status check."""
    index_path = os.path.join(STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "online", "message": "SNJB Institutional RAG API is live."})


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """Processes student questions through the RAG pipeline with safe exception handling."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = query_rag_system(request.query)
        return {"query": request.query, "answer": answer}
    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return JSONResponse(
            status_code=500,
            content={"query": request.query, "answer": f"Service temporarily busy: {str(e)}"}
        )


@app.get("/api/chat/history")
@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str = "default"):
    """Catches all chat history queries cleanly to prevent 404 errors in UI."""
    return {"session_id": session_id, "history": []}


# ----------------- FILE UPLOAD ENDPOINTS -----------------
async def handle_file_upload(file: UploadFile, background_tasks: BackgroundTasks):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    local_path = os.path.join(DOC_FOLDER, file.filename)

    try:
        # Save file to disk
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Index in background task so request returns instantly (prevents web timeouts)
        background_tasks.add_task(index_single_file, local_path, file.filename)

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded successfully! Processing and vector indexing in background."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@app.post("/api/admin/upload")
@app.post("/admin/upload")
@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Handles uploads across all frontend route variations."""
    return await handle_file_upload(file, background_tasks)


# ----------------- DOCUMENT & SYSTEM MANAGEMENT ENDPOINTS -----------------
async def get_document_list():
    """Fetches unique indexed document filenames directly from Qdrant Cloud or local cache."""
    try:
        client = get_qdrant_client()
        res, _ = client.scroll(
            collection_name="institutional_docs",
            limit=100,
            with_payload=True,
            with_vectors=False
        )
        sources = list(set(point.payload.get("source") for point in res if point.payload and "source" in point.payload))
        return {"documents": sources}
    except Exception as e:
        print(f"⚠️ Failed to query Qdrant collection, returning local folder contents: {e}")
        files = []
        if os.path.exists(DOC_FOLDER):
            files = [f for f in os.listdir(DOC_FOLDER) if not f.endswith(".sha256")]
        return {"documents": files}


@app.get("/api/admin/documents")
@app.get("/admin/documents")
async def list_documents():
    """Returns document list for both /api/admin/documents and /admin/documents."""
    return await get_document_list()


@app.get("/api/admin/gaps")
@app.get("/admin/gaps")
async def list_gaps():
    """Returns empty gap list (prevents 404 errors in UI)."""
    return {"gaps": []}


@app.get("/api/health/qdrant")
async def check_qdrant_status():
    """Health check for Qdrant Cloud cluster."""
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name="institutional_docs")
        return {
            "status": "connected",
            "collection": "institutional_docs",
            "vectors_count": info.vectors_count,
            "points_count": info.points_count
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/admin/documents/{filename}")
@app.delete("/admin/documents/{filename}")
async def delete_document(filename: str):
    """Deletes cached document locally."""
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