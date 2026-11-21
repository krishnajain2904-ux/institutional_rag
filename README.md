# Institutional RAG Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers questions from an institution's own documents (prospectus, exam rules, circulars, guidelines) and tells you which document the answer came from. Built for college students and staff.

## Features

- **Multi-format ingestion:** PDF, DOCX, CSV, TXT/MD, and images (OCR via a Groq vision model)
- **Semantic search:** `BAAI/bge-small-en-v1.5` embeddings (FastEmbed) stored in Qdrant
- **Grounded answers:** Llama 3.3 70B (via Groq) answers only from retrieved context, otherwise replies that the information is not in the college records
- **Admin API:** upload documents, list them, preview extracted text, and view logged unanswered questions (documentation gaps)
- **Fast previews:** threadpool-based text extraction for large PDFs
- **Web UI:** static chat page served from `static/`

## Tech stack

Python, FastAPI, LangChain, Qdrant, FastEmbed, Groq, pypdf, python-docx

## Setup

```bash
git clone https://github.com/krishnajain2904-ux/institutional_rag.git
cd institutional_rag
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_key
QDRANT_URL=your_qdrant_cluster_url
QDRANT_API_KEY=your_qdrant_key
```

If `QDRANT_URL`/`QDRANT_API_KEY` are not set, an in-memory Qdrant instance is used (data is lost on restart).

## Run

```bash
uvicorn main:app --reload
```

Open http://localhost:8000 for the chat UI and http://localhost:8000/docs for the API docs.

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Chat UI |
| POST | `/api/chat` | Ask a question (`{"query": "...", "session_id": "..."}`) |
| POST | `/api/admin/upload` | Upload and index a document |
| GET | `/api/admin/documents` | List indexed documents |
| GET | `/api/admin/documents/{filename}/preview` | Preview extracted text |
| GET | `/api/admin/gaps` | Questions the assistant could not answer |

## Project structure

```
main.py           FastAPI app and endpoints
rag_engine.py     Document loading, chunking, embedding, retrieval, LLM answer
auto_crawler.py   Optional script to fetch PDFs from a college web page
static/           Web UI
```

## Limitations

- Unanswered-question log is kept in memory and resets on restart
- No authentication on admin endpoints yet
- No automated tests yet

## License

MIT, see [LICENSE](LICENSE).
