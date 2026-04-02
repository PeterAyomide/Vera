"""CorporateBrain – document ingestion service.

Pipeline: download / receive bytes  →  extract text  →  smart chunk  →  embed  →  store in Supabase.

Key improvements over v1:
  - Markdown tables are kept intact as atomic chunks (never split mid-row).
  - Document title / filename is prepended to every chunk so the LLM always
    knows which document a passage came from.
  - Chunk size reduced to ~1500 chars so granular facts (single-line prices,
    unit counts) land in their own retrievable chunk rather than buried in a
    large block.
  - Overlap increased to 200 chars to reduce boundary misses.
"""

from __future__ import annotations

import io
import logging
import os
from typing import List

import httpx
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from core_engine.services.db import supabase

logger = logging.getLogger(__name__)

# ── OpenAI-compatible client ─────────────────────────────────────────────────
_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


# ── Text extraction ──────────────────────────────────────────────────────────

async def _download_file(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _extract_text_docx(docx_bytes: bytes) -> str:
    """Extract text from .docx preserving paragraph breaks."""
    doc = DocxDocument(io.BytesIO(docx_bytes))
    return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())


def _extract_text_plain(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


# ── Smart chunking ───────────────────────────────────────────────────────────

def _split_preserving_tables(text: str) -> List[str]:
    """
    Split markdown text into segments where markdown tables are kept whole.

    Algorithm:
      1. Walk through lines collecting normal text and table blocks.
      2. When a table block ends, emit it as one segment.
      3. Normal text segments are passed to the recursive splitter later.
    """
    lines = text.split("\n")
    segments: List[str] = []
    current_normal: List[str] = []
    current_table: List[str] = []
    in_table = False

    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    def _flush_normal():
        if current_normal:
            segments.append("\n".join(current_normal).strip())
            current_normal.clear()

    def _flush_table():
        if current_table:
            segments.append("\n".join(current_table).strip())
            current_table.clear()

    for line in lines:
        if _is_table_line(line):
            if not in_table:
                _flush_normal()
                in_table = True
            current_table.append(line)
        else:
            if in_table:
                _flush_table()
                in_table = False
            current_normal.append(line)

    _flush_normal()
    _flush_table()

    return [s for s in segments if s]


def _chunk_text(text: str, filename: str = "") -> List[str]:
    """
    Split text into retrieval-optimised chunks.

    - chunk_size 1500 chars (~300 words) so a single table row, price point,
      or metric lives in its own retrievable chunk.
    - overlap 200 chars catches facts that straddle chunk boundaries.
    - Tables extracted first and kept intact as atomic chunks.
    - Every chunk prefixed with document name for cross-doc synthesis.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Step 1: separate tables from prose
    segments = _split_preserving_tables(text)

    raw_chunks: List[str] = []
    for segment in segments:
        # Keep intact if it's a markdown table and fits in a single chunk
        if "|" in segment and "\n" in segment and len(segment) <= 2000:
            raw_chunks.append(segment)
        else:
            raw_chunks.extend(splitter.split_text(segment))

    # Step 2: prepend document label to every chunk
    doc_label = f"[Document: {filename}]\n\n" if filename else ""
    return [doc_label + chunk for chunk in raw_chunks if chunk.strip()]


# ── Embedding ────────────────────────────────────────────────────────────────

async def _embed_texts(texts: List[str]) -> List[List[float]]:
    response = await _openai.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


# ── Main pipeline: URL-based ingest ─────────────────────────────────────────

async def ingest_document(
    document_id: str,
    file_url: str,
    user_id: str | None = None,
) -> int:
    """Run the full ingest pipeline from a URL. Returns chunks stored."""

    filename = ""
    if not user_id:
        doc_row = (
            supabase.table("documents")
            .select("user_id, title")
            .eq("id", document_id)
            .single()
            .execute()
        )
        user_id = doc_row.data.get("user_id")
        filename = doc_row.data.get("title", "")
        if not user_id:
            raise ValueError(
                f"No user_id supplied and document {document_id} has no user_id."
            )

    logger.info("Downloading document %s from %s", document_id, file_url)
    raw_bytes = await _download_file(file_url)

    ext = os.path.splitext(file_url)[1].lower() or ".docx"
    text = _extract_text_docx(raw_bytes) if ext == ".docx" else _extract_text_plain(raw_bytes)

    if not text.strip():
        raise ValueError(f"Document {document_id} contains no extractable text.")

    chunks = _chunk_text(text, filename=filename)
    logger.info("Document %s split into %d chunks", document_id, len(chunks))

    embeddings = await _embed_texts(chunks)

    rows = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        row = {
            "document_id": document_id,
            "content": chunk,
            "embedding": embedding,
            "chunk_index": idx,
            "metadata": {"filename": filename, "chunk_index": idx},
        }
        if user_id:
            row["user_id"] = user_id
        rows.append(row)

    supabase.table("chunks").insert(rows).execute()
    supabase.table("documents").update({"status": "completed"}).eq("id", document_id).execute()

    logger.info("Document %s ingested successfully (%d chunks)", document_id, len(chunks))
    return len(chunks)


# ── Main pipeline: bytes-based ingest (upload endpoint) ─────────────────────

async def ingest_document_from_bytes(
    document_id: str,
    filename: str,
    file_bytes: bytes,
    user_id: str | None = None,
) -> int:
    """Ingest from raw file bytes (the /upload endpoint). Returns chunks stored."""
    ext = os.path.splitext(filename)[1].lower()

    if ext == ".docx":
        text = _extract_text_docx(file_bytes)
    elif ext in (".txt", ".md"):
        text = _extract_text_plain(file_bytes)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if not text.strip():
        raise ValueError(f"Document '{filename}' contains no extractable text.")

    doc_row: dict = {
        "id": document_id,
        "title": filename,
        "status": "processing",
        "mime_type": {
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".txt": "text/plain",
            ".md": "text/markdown",
        }.get(ext, "application/octet-stream"),
        "file_size": len(file_bytes),
    }
    if user_id:
        doc_row["user_id"] = user_id

    try:
        supabase.table("documents").upsert(doc_row).execute()
    except Exception as e:
        logger.warning("Could not upsert document row for %s: %s – continuing", document_id, e)

    chunks = _chunk_text(text, filename=filename)
    logger.info("Upload '%s' (%s) split into %d chunks", filename, document_id, len(chunks))

    embeddings = await _embed_texts(chunks)

    rows = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        row = {
            "document_id": document_id,
            "content": chunk,
            "embedding": embedding,
            "chunk_index": idx,
            "metadata": {"filename": filename, "chunk_index": idx},
        }
        if user_id:
            row["user_id"] = user_id
        rows.append(row)

    supabase.table("chunks").insert(rows).execute()

    try:
        supabase.table("documents").update({"status": "completed"}).eq("id", document_id).execute()
    except Exception:
        logger.warning("Could not update document status for %s", document_id)

    logger.info("Upload '%s' ingested successfully (%d chunks)", filename, len(chunks))
    return len(chunks)