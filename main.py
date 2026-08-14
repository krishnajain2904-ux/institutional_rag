import os
import shutil
import threading
from datetime import datetime
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

# Supported multi-format file extensions
ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}

# ----------------- IN-MEMORY STATE STORAGE -----------------
CHAT_SESSIONS = {}
KNOWLEDGE_GAPS = [
    {
        "id": 1,
        "query": "Hostel fee structure and mess charges",
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "status": "Pending Analysis"
    },
    {
        "id": 2,
        "query": "Bus route schedule and pickup points",
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "status": "Pending Analysis"
    }
]

# ----------------- BACKGROUND SCHEDULER & LIFESPAN -----------------
scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI Lifespan Manager: Launches crawler sync asynchronously to allow instant port binding."""
    print("🚀 [Startup] Launching initial college portal crawler sync in background thread...")

    # Run initial sync in a daemon thread so server boots immediately (fixes Render deployment timeouts)
    threading.Thread(target=check_and_sync_college_docs, daemon=True).start()

    # Schedule recurring crawler syncs every 6 hours
    scheduler.add_job(check_and_sync_college_docs, "interval", hours=6)
    scheduler.start()

    yield

    scheduler.shutdown()


# Initialize FastAPI Application
app = FastAPI(
    title="SNJB Institutional RAG AI",
    description="Multi-format RAG Assistant & Auto-Crawler for SNJB College of Engineering",
    lifespan=lifespan
)

# Enable CORS for cross-origin frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve Static UI Directory
STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
if os.path.exists(STATIC_FOLDER):
    app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


# ----------------- DATA MODELS -----------------
class QueryRequest(BaseModel):
    query: str
    session_id: str = "session_default"


# ----------------- FRONTEND & CHAT ENDPOINTS -----------------
@app.get("/")
async def serve_frontend():
    """Serves index.html or API status check."""
    index_path = os.path.join(STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "online", "message": "SNJB Institutional RAG API is live."})


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    """Processes student queries through RAG and logs missing info into Knowledge Gaps."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = query_rag_system(request.query)
        session_id = request.session_id

        # 1. Store chat message pair in memory
        if session_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id] = []

        CHAT_SESSIONS[session_id].append({
            "sender": "user",
            "text": request.query,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })
        CHAT_SESSIONS[session_id].append({
            "sender": "bot",
            "text": answer,
            "timestamp": datetime.now().strftime("%I:%M %p")
        })

        # 2. Log unanswered queries as Knowledge Gaps
        if "not available in the college records" in answer.lower():
            KNOWLEDGE_GAPS.append({
                "id": len(KNOWLEDGE_GAPS) + 1,
                "query": request.query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                "status": "Unanswered"
            })

        return {"query": request.query, "answer": answer, "session_id": session_id}
    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return JSONResponse(
            status_code=500,
            content={"query": request.query, "answer": f"Service temporarily busy: {str(e)}"}
        )


@app.get("/api/chat/history")
@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str = "session_default"):
    """Returns active session chat history to UI."""
    history = CHAT_SESSIONS.get(session_id, [])
    return {"session_id": session_id, "history": history, "messages": history}


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
        # Save file to server disk
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Index vectors asynchronously using background tasks
        background_tasks.add_task(index_single_file, local_path, file.filename)

        file_size_bytes = os.path.getsize(local_path)
        file_size_str = f"{round(file_size_bytes / 1024, 1)} KB" if file_size_bytes < 1048576 else f"{round(file_size_bytes / 1048576, 2)} MB"

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded successfully!",
            "document": {
                "filename": file.filename,
                "name": file.filename,
                "upload_date": datetime.now().strftime("%Y-%m-%d"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "size": file_size_str,
                "chunks": "Indexing..."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.post("/api/admin/upload")
@app.post("/admin/upload")
@app.post("/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Handles uploads across all frontend route variations."""
    return await handle_file_upload(file, background_tasks)


# ----------------- DOCUMENT & SYSTEM MANAGEMENT ENDPOINTS -----------------
async def get_document_list():
    """Queries Qdrant Cloud and returns structured document objects for the frontend UI table."""
    formatted_docs = []

    try:
        client = get_qdrant_client()
        res, _ = client.scroll(
            collection_name="institutional_docs",
            limit=100,
            with_payload=True,
            with_vectors=False
        )

        # Count chunks per source document
        source_counts = {}
        for point in res:
            if point.payload and "source" in point.payload:
                src = point.payload["source"]
                source_counts[src] = source_counts.get(src, 0) + 1

        for filename, chunk_count in source_counts.items():
            local_file = os.path.join(DOC_FOLDER, filename)
            size_str = "Cloud Sync"
            if os.path.exists(local_file):
                b = os.path.getsize(local_file)
                size_str = f"{round(b / 1024, 1)} KB" if b < 1048576 else f"{round(b / 1048576, 2)} MB"

            formatted_docs.append({
                "filename": filename,
                "name": filename,
                "title": filename,
                "upload_date": datetime.now().strftime("%Y-%m-%d"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "size": size_str,
                "chunks": chunk_count,
                "status": "Indexed"
            })

    except Exception as e:
        print(f"⚠️ Falling back to local documents folder: {e}")
        if os.path.exists(DOC_FOLDER):
            for filename in os.listdir(DOC_FOLDER):
                if not filename.endswith(".sha256"):
                    file_path = os.path.join(DOC_FOLDER, filename)
                    b = os.path.getsize(file_path)
                    size_str = f"{round(b / 1024, 1)} KB" if b < 1048576 else f"{round(b / 1048576, 2)} MB"
                    formatted_docs.append({
                        "filename": filename,
                        "name": filename,
                        "title": filename,
                        "upload_date": datetime.now().strftime("%Y-%m-%d"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "size": size_str,
                        "chunks": "Local",
                        "status": "Indexed"
                    })

    return {"documents": formatted_docs, "files": formatted_docs, "data": formatted_docs}


@app.get("/api/admin/documents")
@app.get("/admin/documents")
async def list_documents():
    """Returns document list for both /api/admin/documents and /admin/documents."""
    return await get_document_list()


@app.get("/api/admin/gaps")
@app.get("/admin/gaps")
async def list_gaps():
    """Returns logged unanswered student queries for the Knowledge Gaps tab."""
    return {"gaps": KNOWLEDGE_GAPS, "data": KNOWLEDGE_GAPS}


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
    """Deletes document locally."""
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


# ----------------- DIRECT SESSION CATCH-ALL ROUTE -----------------
@app.get("/{session_id}")
async def catch_direct_session(session_id: str):
    """Catches direct session ID lookups (e.g. GET /session_b7vzq23) to prevent 404 errors."""
    if session_id.startswith("session_"):
        history = CHAT_SESSIONS.get(session_id, [])
        return {"session_id": session_id, "history": history, "messages": history}

    # Return 404 for actual unknown web paths
    raise HTTPException(status_code=404, detail="Resource not found")