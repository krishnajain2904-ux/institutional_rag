import os
import shutil
import time
import threading
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel

from rag_engine import (
    query_rag_system,
    index_single_file,
    DOC_FOLDER
)

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".csv", ".pptx", ".ppt",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp"
}

KNOWLEDGE_GAPS = []

app = FastAPI(
    title="InstiAssistant RAG Engine",
    description="Campus Knowledge Assistant API"
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


def extract_preview_text(file_path: str, max_chars: int = 3000) -> str:
    """Fast, lightweight text extractor specifically optimized for instant previews."""
    ext = os.path.splitext(file_path)[1].lower()
    try:
        if ext == ".pdf":
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            extracted = []
            total_len = 0
            for page in reader.pages[:10]:  # Only inspect up to first 10 pages
                text = page.extract_text() or ""
                if text.strip():
                    extracted.append(text)
                    total_len += len(text)
                if total_len >= max_chars:
                    break
            return "\n\n".join(extracted)[:max_chars] if extracted else "No extractable text found in this PDF."

        elif ext in [".docx", ".doc"]:
            import docx
            doc_obj = docx.Document(file_path)
            text_list = []
            total_len = 0
            for p in doc_obj.paragraphs:
                if p.text.strip():
                    text_list.append(p.text)
                    total_len += len(p.text)
                    if total_len >= max_chars:
                        break
            return "\n\n".join(text_list)[:max_chars] if text_list else "No text found."

        elif ext in [".txt", ".md", ".csv"]:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read(max_chars)
        else:
            return f"Preview not supported for '{ext}' files."
    except Exception as e:
        return f"Error reading preview: {e}"


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(STATIC_FOLDER, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return JSONResponse({"status": "online", "message": "InstiAssistant API is live."})


@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        answer = query_rag_system(request.query)
        session_id = request.session_id

        if "not available in the college records" in answer.lower():
            KNOWLEDGE_GAPS.append({
                "id": len(KNOWLEDGE_GAPS) + 1,
                "query": request.query,
                "question": request.query,
                "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                "date": datetime.now().strftime("%Y-%m-%d")
            })

        return {"query": request.query, "answer": answer, "session_id": session_id}
    except Exception as e:
        print(f"❌ Error in /api/chat: {e}")
        return JSONResponse(
            status_code=200,
            content={
                "query": request.query,
                "answer": "This information is not available in the college records.",
                "session_id": request.session_id
            }
        )


@app.post("/api/admin/upload")
async def upload_document(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'.")

    local_path = os.path.join(DOC_FOLDER, file.filename)
    try:
        with open(local_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        background_tasks.add_task(index_single_file, local_path, file.filename)

        file_size_bytes = os.path.getsize(local_path)
        file_size_str = (
            f"{round(file_size_bytes / 1024, 1)} KB"
            if file_size_bytes < 1048576
            else f"{round(file_size_bytes / 1048576, 2)} MB"
        )

        return {
            "status": "success",
            "filename": file.filename,
            "message": f"File '{file.filename}' uploaded successfully!",
            "document": {
                "filename": file.filename,
                "upload_date": datetime.now().strftime("%Y-%m-%d"),
                "size": file_size_str,
                "chunks": "Indexing..."
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")


@app.get("/api/admin/documents")
async def list_documents():
    docs_list = []
    if os.path.exists(DOC_FOLDER):
        for filename in sorted(os.listdir(DOC_FOLDER)):
            if filename.startswith(".") or filename.endswith(".sha256"):
                continue
            file_path = os.path.join(DOC_FOLDER, filename)
            if os.path.isfile(file_path):
                b = os.path.getsize(file_path)
                size_str = (
                    f"{round(b / 1024, 1)} KB"
                    if b < 1048576
                    else f"{round(b / 1048576, 2)} MB"
                )
                mtime = os.path.getmtime(file_path)
                docs_list.append({
                    "filename": filename,
                    "upload_date": datetime.fromtimestamp(mtime).strftime("%Y-%m-%d"),
                    "size": size_str,
                    "chunks": "Indexed",
                    "status": "Ready"
                })
    return {"documents": docs_list}


@app.get("/api/admin/documents/{filename}/preview")
async def preview_document(filename: str):
    file_path = os.path.join(DOC_FOLDER, filename)
    if os.path.exists(file_path):
        preview_content = await run_in_threadpool(extract_preview_text, file_path)
        return {
            "filename": filename,
            "preview_content": preview_content,
            "source": "Local File"
        }

    raise HTTPException(status_code=404, detail="File content not found.")


@app.get("/api/admin/gaps")
async def list_gaps():
    return {"gaps": KNOWLEDGE_GAPS}