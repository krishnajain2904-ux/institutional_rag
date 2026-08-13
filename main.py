import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_engine import (
    index_single_pdf,
    answer_query,
    get_all_documents,
    delete_document,
    get_knowledge_gaps,
    resolve_gap,
    get_session_history_list,
    DOC_FOLDER
)

app = FastAPI(
    title="Institutional Knowledge Assistant",
    description="Backend API for RAG-based Institutional Knowledge Base",
    version="2.5.0"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs(DOC_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default_session"


# ----------------- CHAT API ENDPOINTS -----------------

@app.post("/api/chat")
async def chat_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    answer, sources = answer_query(request.query, session_id=request.session_id)
    return {
        "answer": answer,
        "sources": sources
    }


@app.get("/api/chat/history/{session_id}")
async def fetch_chat_history(session_id: str):
    history = get_session_history_list(session_id)
    return {"history": history}


# ----------------- ADMIN API ENDPOINTS -----------------

@app.post("/api/admin/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")

    file_path = os.path.join(DOC_FOLDER, file.filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunk_count = index_single_pdf(file_path, file.filename)
        return {
            "message": f"Successfully indexed '{file.filename}'",
            "filename": file.filename,
            "chunks": chunk_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to index PDF: {str(e)}")


@app.get("/api/admin/documents")
async def list_documents():
    docs = get_all_documents()
    return {"documents": docs}


@app.get("/api/admin/documents/view/{filename}")
async def view_document(filename: str):
    file_path = os.path.join(DOC_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Requested file not found.")
    return FileResponse(file_path, media_type="application/pdf")


@app.delete("/api/admin/documents/{filename}")
async def remove_document(filename: str):
    try:
        delete_document(filename)
        return {"message": f"Successfully deleted '{filename}' record."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


# ----------------- KNOWLEDGE GAP ENDPOINTS -----------------

@app.get("/api/admin/gaps")
async def fetch_knowledge_gaps():
    gaps = get_knowledge_gaps()
    return {"gaps": gaps}


@app.post("/api/admin/gaps/{gap_id}/resolve")
async def mark_gap_resolved(gap_id: int):
    resolve_gap(gap_id)
    return {"message": f"Gap #{gap_id} marked as resolved."}


# Mount Static Files UI
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)