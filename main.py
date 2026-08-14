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

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}

CHAT_SESSIONS = {}
KNOWLEDGE_GAPS = [
    {
        "id": 1,
        "query": "Hostel fee structure and mess charges for 2026",
        "question": "Hostel fee structure and mess charges for 2026",
        "topic": "Hostel & Fees",
        "category": "Admissions / Fees",
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "Pending Analysis",
        "count": 1
    },
    {
        "id": 2,
        "query": "Bus route schedule and pickup points from Jalgaon",
        "question": "Bus route schedule and pickup points from Jalgaon",
        "topic": "Transportation",
        "category": "Campus Logistics",
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "status": "Pending Analysis",
        "count": 1
    }
]

scheduler = BackgroundScheduler()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 [Startup] Launching initial college portal crawler sync in background thread...")
    threading.Thread(target=check_and_sync_college_docs, daemon=True).start()

    scheduler.add_job(check_and_sync_college_docs, "interval", hours=6)
    scheduler.start()

    yield
    scheduler.shutdown()


app = FastAPI(
    title="SNJB Institutional RAG AI",
    description="Multi-format RAG Assistant & Auto-Crawler for SNJB College of Engineering",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_FOLDER = os.path.abspath(os.path.join(os.path.dirname(__file__), "static"))
if os.path.exists(STATIC_FOLDER):
    app.mount("/static", StaticFiles(directory=STATIC_FOLDER), name="static")


class QueryRequest(BaseModel):
    query: str
    session_id: str = "session_default"


class TextPasteRequest(BaseModel):
    title: str
    content: str


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "online", "message": "SNJB Institutional RAG API is live."})


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = query_rag_system(request.query)
        session_id = request.session_id

        if session_id not in CHAT_SESSIONS:
            CHAT_SESSIONS[session_id] = []

        now_time = datetime.now().strftime("%I:%M %p")
        CHAT_SESSIONS[session_id].append({"sender": "user", "text": request.query, "timestamp": now_time})
        CHAT_SESSIONS[session_id].append({"sender": "bot", "text": answer, "timestamp": now_time})

        if "not available in the college records" in answer.lower():
            now_str = datetime.now().strftime("%Y-%m-%d %I:%M %p")
            KNOWLEDGE_GAPS.append({
                "id": len(KNOWLEDGE_GAPS) + 1,
                "query": request.query,
                "question": request.query,
                "topic": "Unanswered Query",
                "category": "General",
                "timestamp": now_str,
                "date": datetime.now().strftime("%Y-%m-%d"),
                "status": "Unanswered",
                "count": 1
            })

        return {"query": request.query, "answer": answer, "session_id": session_id}
    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return JSONResponse(
            status_code=200,
            content={"query": request.query, "answer": "This information is not available in the college records.",
                     "session_id": request.session_id}
        )


@app.get("/api/chat/history")
@app.get("/api/chat/history/{session_id}")
async def get_chat_history(session_id: str = "session_default"):
    history = CHAT_SESSIONS.get(session_id, [])
    return {"session_id": session_id, "history": history, "messages": history}


async def handle_file_upload(file: UploadFile, background_tasks: BackgroundTasks):
    ext = os.path.splitext(file.filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed extensions: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    local_path = os.path.join(DOC_FOLDER, file.filename)

    try:
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(index_single_file, local_path, file.filename)

        file_size_bytes = os.path.getsize(local_path)
        file_size_str = f"{round(file_size_bytes / 1024, 1)} KB" if file_size_bytes < 1048576 else f"{round(file_size_bytes / 1048576, 2)} MB"

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded and saved successfully!",
            "document": {
                "filename": file.filename,
                "name": file.filename,
                "title": file.filename,
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
    return await handle_file_upload(file, background_tasks)


@app.post("/api/admin/paste-text")
@app.post("/admin/paste-text")
async def paste_text_endpoint(request: TextPasteRequest, background_tasks: BackgroundTasks):
    if not request.content.strip():
        raise HTTPException(status_code=400, detail="Pasted text content cannot be empty.")

    clean_title = request.title.strip().replace(" ", "_") or f"Notice_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if not clean_title.endswith(".txt"):
        clean_title += ".txt"

    local_path = os.path.join(DOC_FOLDER, clean_title)

    try:
        with open(local_path, "w", encoding="utf-8") as f:
            f.write(request.content)

        background_tasks.add_task(index_single_file, local_path, clean_title)

        file_size_bytes = os.path.getsize(local_path)
        file_size_str = f"{round(file_size_bytes / 1024, 1)} KB"

        return {
            "status": "success",
            "filename": clean_title,
            "message": f"Text notice '{clean_title}' saved and indexed successfully!",
            "document": {
                "filename": clean_title,
                "name": clean_title,
                "title": clean_title,
                "upload_date": datetime.now().strftime("%Y-%m-%d"),
                "date": datetime.now().strftime("%Y-%m-%d"),
                "size": file_size_str,
                "chunks": "Indexing..."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save text: {str(e)}")


async def get_document_list():
    docs_map = {}

    # 1. Scan local documents folder so uploaded files show instantly
    if os.path.exists(DOC_FOLDER):
        for filename in os.listdir(DOC_FOLDER):
            if not filename.endswith(".sha256"):
                file_path = os.path.join(DOC_FOLDER, filename)
                if os.path.isfile(file_path):
                    b = os.path.getsize(file_path)
                    size_str = f"{round(b / 1024, 1)} KB" if b < 1048576 else f"{round(b / 1048576, 2)} MB"
                    docs_map[filename] = {
                        "filename": filename,
                        "name": filename,
                        "title": filename,
                        "upload_date": datetime.now().strftime("%Y-%m-%d"),
                        "date": datetime.now().strftime("%Y-%m-%d"),
                        "size": size_str,
                        "chunks": "Local File",
                        "status": "Ready"
                    }

    # 2. Merge with Qdrant vector store point counts
    try:
        client = get_qdrant_client()
        res, _ = client.scroll(
            collection_name="institutional_docs",
            limit=200,
            with_payload=True,
            with_vectors=False
        )

        source_counts = {}
        for point in res:
            if point.payload and "source" in point.payload:
                src = point.payload["source"]
                source_counts[src] = source_counts.get(src, 0) + 1

        for filename, chunk_count in source_counts.items():
            if filename in docs_map:
                docs_map[filename]["chunks"] = f"{chunk_count} Chunks"
                docs_map[filename]["status"] = "Indexed"
            else:
                docs_map[filename] = {
                    "filename": filename,
                    "name": filename,
                    "title": filename,
                    "upload_date": datetime.now().strftime("%Y-%m-%d"),
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "size": "Cloud Sync",
                    "chunks": f"{chunk_count} Chunks",
                    "status": "Indexed"
                }

    except Exception as e:
        print(f"⚠️ Qdrant scroll warning: {e}")

    formatted_docs = list(docs_map.values())
    return {"documents": formatted_docs, "files": formatted_docs, "data": formatted_docs}


@app.get("/api/admin/documents")
@app.get("/admin/documents")
async def list_documents():
    return await get_document_list()


@app.get("/api/admin/gaps")
@app.get("/admin/gaps")
async def list_gaps():
    return {
        "gaps": KNOWLEDGE_GAPS,
        "data": KNOWLEDGE_GAPS,
        "items": KNOWLEDGE_GAPS
    }


@app.get("/api/health/qdrant")
async def check_qdrant_status():
    try:
        client = get_qdrant_client()
        info = client.get_collection(collection_name="institutional_docs")
        points = getattr(info, "points_count", getattr(info, "vectors_count", 0))
        return {
            "status": "connected",
            "collection": "institutional_docs",
            "points_count": points,
            "vectors_count": points
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.delete("/api/admin/documents/{filename}")
@app.delete("/admin/documents/{filename}")
async def delete_document(filename: str):
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


@app.get("/{session_id}")
async def catch_direct_session(session_id: str):
    if session_id.startswith("session_"):
        history = CHAT_SESSIONS.get(session_id, [])
        return {"session_id": session_id, "history": history, "messages": history}
    raise HTTPException(status_code=404, detail="Resource not found")