"""Vera — AI operating system for digital agencies.

Three capabilities:
  1. CorporateBrain  — upload docs, query via RAG
  2. LeadSentinel    — score leads, manage pipeline
  3. Outreach Engine — personalized email drafts from knowledge + lead data

Dashboard served at /
"""

from __future__ import annotations

import collections
import logging
import os
import time
import uuid
from typing import Deque, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── Services ──────────────────────────────────────────────────────────────────
from services.ingest_service import ingest_document_from_bytes
from services.query_service import query_knowledge
from services.lead_service import analyze_lead
from services.lead_dashboard_service import (
    list_leads, get_lead, update_lead, get_dashboard_stats,
)
from services.auth import require_api_key, sanitize_search_input
from services.chat_service import (
    create_session, list_sessions, delete_session,
    save_message, load_messages, update_session_title,
)
from services.db import supabase

# ── Pydantic models ───────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    user_id: Optional[str] = None
    top_k: int = 5
    session_id: Optional[str] = None

class LeadAnalyzeRequest(BaseModel):
    lead_id: str
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    message: Optional[str] = None
    user_id: Optional[str] = None

class LeadListQuery(BaseModel):
    user_id: Optional[str] = None
    status: Optional[str] = None
    priority_min: Optional[int] = None
    score_min: Optional[float] = None
    search: Optional[str] = None
    sort_by: str = "created_at"
    sort_order: str = "desc"
    page: int = 1
    page_size: int = 50

class LeadUpdateRequest(BaseModel):
    status: Optional[str] = None
    notes: Optional[str] = None
    priority: Optional[int] = None

class CreateSessionRequest(BaseModel):
    title: Optional[str] = "New conversation"

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Vera",
    version="1.0.0",
    description="AI operating system for digital agencies.",
)

# ── Rate limiting ─────────────────────────────────────────────────────────────
# Sliding-window in-memory rate limiter.
# Limits are per (api-key + path-prefix) per 60 seconds.

_RATE_LIMITS: Dict[str, int] = {
    "/query":        int(os.environ.get("RATE_LIMIT_QUERY",    "30")),
    "/analyze-lead": int(os.environ.get("RATE_LIMIT_ANALYZE",  "30")),
    "/upload":       int(os.environ.get("RATE_LIMIT_UPLOAD",   "10")),
    "/leads":        int(os.environ.get("RATE_LIMIT_LEADS",    "60")),
    "/chat":         int(os.environ.get("RATE_LIMIT_CHAT",     "60")),
}
_DEFAULT_RATE  = int(os.environ.get("RATE_LIMIT_DEFAULT", "120"))
_WINDOW        = 60  # seconds

_buckets: Dict[str, Deque[float]] = collections.defaultdict(collections.deque)

_SKIP_RATE_LIMIT = {"/health", "/", "/dashboard", "/embed.js"}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    path = request.url.path
    if path in _SKIP_RATE_LIMIT or path.startswith("/static"):
        return await call_next(request)

    api_key   = request.headers.get("X-API-Key", "")
    client_ip = request.client.host if request.client else "unknown"
    prefix    = path.split("/")[1] if "/" in path[1:] else path.lstrip("/")
    key       = f"{api_key or client_ip}:{prefix}"

    limit = next(
        (v for k, v in _RATE_LIMITS.items() if path.startswith(k)),
        _DEFAULT_RATE,
    )

    now    = time.monotonic()
    bucket = _buckets[key]
    while bucket and bucket[0] < now - _WINDOW:
        bucket.popleft()

    if len(bucket) >= limit:
        retry = int(_WINDOW - (now - bucket[0])) + 1
        return JSONResponse(
            status_code=429,
            content={"detail": "Rate limit exceeded. Please slow down."},
            headers={"Retry-After": str(retry)},
        )

    bucket.append(now)
    return await call_next(request)

# ── CORS ──────────────────────────────────────────────────────────────────────

_env_origins = os.environ.get("ALLOWED_ORIGINS", "")
ORIGINS = (
    [o.strip() for o in _env_origins.split(",") if o.strip()]
    or ["http://localhost:8000", "http://127.0.0.1:8000", "http://localhost:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    allow_credentials=True,
)

# ─── Health & UI ──────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "product": "Vera", "version": "1.0.0"}

@app.get("/")
@app.get("/dashboard")
async def serve_dashboard():
    return FileResponse("static/dashboard.html")

@app.get("/embed.js")
async def serve_embed():
    return FileResponse("static/embed.js", media_type="application/javascript")

# ─── CorporateBrain: Upload & Ingest ──────────────────────────────────────────

ALLOWED_EXTS = {".docx", ".txt", ".md", ".pdf"}

@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    user_id: str = Form(None),
    _key: str = Depends(require_api_key),
):
    """Upload and index a document into the knowledge base."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Use: {', '.join(sorted(ALLOWED_EXTS))}")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "File is empty.")

    doc_id = str(uuid.uuid4())
    try:
        chunks = await ingest_document_from_bytes(
            document_id=doc_id,
            filename=file.filename,
            file_bytes=raw,
            user_id=user_id,
        )
        logger.info("Indexed '%s' → %d chunks", file.filename, chunks)
        return {"document_id": doc_id, "filename": file.filename, "chunks_stored": chunks}
    except Exception as e:
        logger.exception("Ingest failed: %s", file.filename)
        raise HTTPException(500, "Ingestion failed.") from e


@app.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    _key: str = Depends(require_api_key),
):
    """Delete an indexed document and its chunks."""
    try:
        result = (
            supabase.table("documents")
            .delete()
            .eq("id", document_id)
            .execute()
        )
        if not result.data:
            raise HTTPException(404, "Document not found.")
        return {"deleted": True, "document_id": document_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Document delete failed: %s", document_id)
        raise HTTPException(500, "Document deletion failed.") from e

# ─── CorporateBrain: RAG Query ────────────────────────────────────────────────

@app.post("/query")
async def query(payload: QueryRequest, _key: str = Depends(require_api_key)):
    """Query the knowledge base. Pass session_id for conversational memory."""
    try:
        result = await query_knowledge(
            question=payload.question,
            user_id=payload.user_id,
            top_k=payload.top_k,
            session_id=payload.session_id,
        )
        answer  = result["answer"]
        sources = result.get("sources", [])

        # Persist messages to Supabase if a session_id was provided
        if payload.session_id:
            import asyncio
            asyncio.create_task(_persist_messages(
                payload.session_id, payload.question, answer, sources
            ))

        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.exception("Query failed: %s", payload.question[:80])
        raise HTTPException(500, "Query failed.") from e

async def _persist_messages(session_id, question, answer, sources):
    """Save user + assistant messages in background (non-blocking)."""
    try:
        # Set session title from first user message
        msgs = load_messages(session_id)
        if not msgs:
            update_session_title(session_id, question[:80])

        save_message(session_id, "user",      question, [])
        save_message(session_id, "assistant", answer,   sources)
    except Exception as e:
        logger.warning("Message persistence failed: %s", e)

# ─── Chat Sessions ────────────────────────────────────────────────────────────

@app.post("/chat/sessions")
async def new_chat_session(
    payload: CreateSessionRequest,
    _key: str = Depends(require_api_key),
):
    """Create a new chat session."""
    session = create_session(payload.title or "New conversation")
    return session

@app.get("/chat/sessions")
async def get_chat_sessions(_key: str = Depends(require_api_key)):
    """List recent chat sessions."""
    return {"sessions": list_sessions()}

@app.get("/chat/sessions/{session_id}/messages")
async def get_messages(session_id: str, _key: str = Depends(require_api_key)):
    """Load message history for a session."""
    return {"messages": load_messages(session_id)}

@app.delete("/chat/sessions/{session_id}")
async def del_session(session_id: str, _key: str = Depends(require_api_key)):
    """Delete a chat session and all its messages."""
    delete_session(session_id)
    return {"deleted": True}

# ─── LeadSentinel: Score ──────────────────────────────────────────────────────

@app.post("/analyze-lead")
async def score_lead(payload: LeadAnalyzeRequest, _key: str = Depends(require_api_key)):
    """Score an inbound lead with AI and persist to Supabase."""
    try:
        scoring = await analyze_lead(
            lead_id=payload.lead_id,
            name=payload.name,
            email=payload.email or "",
            company=payload.company or "",
            message=payload.message or "",
            user_id=payload.user_id,
        )
        return {"lead_id": payload.lead_id, "scoring": scoring.model_dump()}
    except Exception as e:
        logger.exception("Lead scoring failed: %s", payload.lead_id)
        raise HTTPException(500, "Scoring failed.") from e

# ─── LeadSentinel: Pipeline ───────────────────────────────────────────────────

@app.post("/leads")
async def get_leads(payload: LeadListQuery, _key: str = Depends(require_api_key)):
    """List leads with optional filters and pagination."""
    try:
        return list_leads(
            user_id=payload.user_id,
            status=payload.status,
            priority_min=payload.priority_min,
            score_min=payload.score_min,
            search=sanitize_search_input(payload.search) if payload.search else None,
            sort_by=payload.sort_by,
            sort_order=payload.sort_order,
            page=payload.page,
            page_size=payload.page_size,
        )
    except Exception as e:
        logger.exception("List leads failed")
        raise HTTPException(500, "Failed.") from e

@app.get("/leads/{lead_id}")
async def get_lead_ep(lead_id: str, _key: str = Depends(require_api_key)):
    lead = get_lead(lead_id)
    if not lead:
        raise HTTPException(404, "Lead not found.")
    return {"lead": lead}

@app.patch("/leads/{lead_id}")
async def update_lead_ep(
    lead_id: str,
    payload: LeadUpdateRequest,
    _key: str = Depends(require_api_key),
):
    updates = payload.model_dump(exclude_none=True)
    if not updates:
        raise HTTPException(400, "Nothing to update.")
    lead = update_lead(lead_id, updates)
    if not lead:
        raise HTTPException(404, "Lead not found.")
    return {"lead": lead}

@app.get("/dashboard/stats")
async def stats(
    user_id: Optional[str] = None,
    _key: str = Depends(require_api_key),
):
    try:
        return get_dashboard_stats(user_id=user_id)
    except Exception as e:
        logger.exception("Stats failed")
        raise HTTPException(500, "Failed.") from e
