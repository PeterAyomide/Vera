"""CorporateBrain – document ingestion service.

Pipeline: download / receive bytes  →  extract text  →  clean  →  smart chunk
          →  embed  →  store in Supabase.

Supported file types:
  .docx  — paragraphs + tables + footnotes + text boxes
  .pdf   — pdfplumber (layout-aware) with pytesseract OCR fallback for scans
  .xlsx  — every sheet as labelled rows
  .pptx  — slide text + speaker notes
  .txt / .md — plain UTF-8

Key design rules:
  - Markdown tables kept intact as atomic chunks (never split mid-row).
  - Document title / filename prepended to every chunk for cross-doc synthesis.
  - Chunk size 1500 chars so granular facts land in their own retrievable chunk.
  - Overlap 200 chars catches facts that straddle chunk boundaries.
  - Text cleaned before chunking: artifacts, watermarks, rogue page numbers removed.
"""

from __future__ import annotations

import io
import logging
import os
import re
from typing import List

import httpx
from docx import Document as DocxDocument
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import AsyncOpenAI

from services.db import supabase

logger = logging.getLogger(__name__)

# ── OpenAI-compatible client ─────────────────────────────────────────────────
_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# Repeated watermark words that pollute chunks if left in
_WATERMARK_WORDS = {"CONFIDENTIAL", "DRAFT", "PRIVILEGED", "SAMPLE", "COPY"}


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """Normalise extracted text before chunking.

    Removes common extraction artifacts:
      - Tab characters collapsed to single space
      - Multiple consecutive spaces collapsed
      - Isolated page numbers (lone digits on their own line)
      - Repeated watermark words (e.g. CONFIDENTIAL CONFIDENTIAL CONFIDENTIAL)
      - Windows-style line endings normalised
      - Excessive blank lines (3+ collapsed to 2)
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\t+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
    for word in _WATERMARK_WORDS:
        text = re.sub(rf"(\b{word}\b\s*){{2,}}", f"{word} ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── Text extraction ───────────────────────────────────────────────────────────

async def _download_file(url: str) -> bytes:
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content


def _extract_text_docx(docx_bytes: bytes) -> str:
    """Extract text from .docx: paragraphs + tables + footnotes + text boxes.

    Walks the document body XML in document order so content appears in the
    correct reading sequence. Also surfaces footnote text (common in legal
    documents) and text boxes which python-docx skips by default.
    """
    from docx.oxml.ns import qn

    doc = DocxDocument(io.BytesIO(docx_bytes))
    parts: List[str] = []

    # ── Body: paragraphs and tables in document order ────────────────────────
    for block in doc.element.body:
        tag = block.tag

        if tag == qn("w:p"):
            text = "".join(
                node.text or ""
                for node in block.iter()
                if node.tag == qn("w:t")
            )
            if text.strip():
                parts.append(text.strip())

        elif tag == qn("w:tbl"):
            # Emit each row as pipe-delimited markdown so the table splitter
            # keeps it atomic.
            for row in block.findall(".//" + qn("w:tr")):
                cells: List[str] = []
                for cell in row.findall(".//" + qn("w:tc")):
                    cell_text = "".join(
                        node.text or ""
                        for node in cell.iter()
                        if node.tag == qn("w:t")
                    ).strip()
                    cells.append(cell_text)
                if any(cells):
                    parts.append("| " + " | ".join(cells) + " |")

    # ── Footnotes ────────────────────────────────────────────────────────────
    try:
        footnotes_part = doc.part.footnotes
        if footnotes_part is not None:
            fn_root = footnotes_part._element
            for fn in fn_root.findall(qn("w:footnote")):
                fn_id = fn.get(qn("w:id"), "")
                if fn_id in ("-1", "0"):
                    continue
                fn_text = "".join(
                    node.text or ""
                    for node in fn.iter()
                    if node.tag == qn("w:t")
                ).strip()
                if fn_text:
                    parts.append(f"[Footnote] {fn_text}")
    except Exception as exc:
        logger.debug("Footnote extraction skipped: %s", exc)

    return _clean_text("\n\n".join(parts))


def _extract_text_plain(raw: bytes) -> str:
    return _clean_text(raw.decode("utf-8", errors="replace"))


def _extract_text_pdf(pdf_bytes: bytes) -> str:
    """Extract text from a PDF.

    Strategy:
      1. pdfplumber — layout-aware, preserves table structure and handles
         multi-column documents better than pypdf.
      2. If pdfplumber returns fewer than 100 chars total the PDF is likely a
         scanned image. Fall back to pytesseract OCR via pdf2image.
      3. If OCR dependencies are unavailable a clear warning is logged and an
         empty string returned so the caller surfaces a friendly error.
    """
    try:
        import pdfplumber
    except ImportError:
        raise ImportError(
            "pdfplumber is required for PDF extraction. "
            "Add 'pdfplumber' to requirements.txt."
        )

    pages: List[str] = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_parts: List[str] = []

            # Tables first — extract as markdown rows
            for table in page.extract_tables():
                for row in table:
                    cells = [str(cell or "").strip() for cell in row]
                    if any(cells):
                        page_parts.append("| " + " | ".join(cells) + " |")

            # Remaining prose text
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_parts.append(page_text.strip())

            if page_parts:
                pages.append("\n".join(page_parts))

    full_text = "\n\n".join(pages)

    # OCR fallback for scanned PDFs
    if len(full_text.strip()) < 100:
        logger.info("PDF text layer sparse — attempting OCR fallback")
        full_text = _ocr_pdf(pdf_bytes)

    return _clean_text(full_text)


def _ocr_pdf(pdf_bytes: bytes) -> str:
    """Rasterise PDF pages and run Tesseract OCR on each.

    Requires: pdf2image (wraps poppler) and pytesseract + system Tesseract.
    Returns empty string gracefully if either dependency is missing.
    """
    try:
        from pdf2image import convert_from_bytes
        import pytesseract
    except ImportError:
        logger.warning(
            "OCR dependencies not installed (pdf2image / pytesseract). "
            "Scanned PDF will produce no text. "
            "Install with: pip install pdf2image pytesseract"
        )
        return ""

    try:
        images = convert_from_bytes(pdf_bytes, dpi=200)
        pages = [pytesseract.image_to_string(img) for img in images]
        return "\n\n".join(p.strip() for p in pages if p.strip())
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return ""


def _extract_text_xlsx(xlsx_bytes: bytes) -> str:
    """Extract text from an Excel workbook.

    Each sheet is emitted as a labelled markdown table so the chunker keeps
    rows intact. Empty sheets are skipped.
    """
    try:
        import openpyxl
    except ImportError:
        raise ImportError(
            "openpyxl is required for Excel extraction. "
            "Add 'openpyxl' to requirements.txt."
        )

    wb = openpyxl.load_workbook(io.BytesIO(xlsx_bytes), read_only=True, data_only=True)
    sheets: List[str] = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows: List[str] = []

        for row in ws.iter_rows(values_only=True):
            cells = [str(cell) if cell is not None else "" for cell in row]
            if not any(c.strip() for c in cells):
                continue
            rows.append("| " + " | ".join(cells) + " |")

        if rows:
            sheets.append(f"## Sheet: {sheet_name}\n\n" + "\n".join(rows))

    wb.close()
    return _clean_text("\n\n".join(sheets))


def _extract_text_pptx(pptx_bytes: bytes) -> str:
    """Extract text from a PowerPoint presentation.

    Each slide is emitted as a labelled section containing:
      - All text from shapes (titles, body text, text boxes)
      - Speaker notes (often contain important context)
    """
    try:
        from pptx import Presentation
    except ImportError:
        raise ImportError(
            "python-pptx is required for PowerPoint extraction. "
            "Add 'python-pptx' to requirements.txt."
        )

    prs = Presentation(io.BytesIO(pptx_bytes))
    slides: List[str] = []

    for i, slide in enumerate(prs.slides, 1):
        slide_parts: List[str] = [f"## Slide {i}"]

        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                text = "".join(run.text for run in para.runs).strip()
                if text:
                    slide_parts.append(text)

        if slide.has_notes_slide:
            notes_frame = slide.notes_slide.notes_text_frame
            notes_text = "\n".join(
                para.text.strip()
                for para in notes_frame.paragraphs
                if para.text.strip()
            )
            if notes_text:
                slide_parts.append(f"[Speaker Notes] {notes_text}")

        if len(slide_parts) > 1:
            slides.append("\n".join(slide_parts))

    return _clean_text("\n\n".join(slides))


# ── Smart chunking ────────────────────────────────────────────────────────────

def _split_preserving_tables(text: str) -> List[str]:
    """Split text into segments where markdown tables are kept whole."""
    lines = text.split("\n")
    segments: List[str] = []
    current_normal: List[str] = []
    current_table: List[str] = []
    in_table = False

    def _is_table_line(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|")

    def _flush_normal() -> None:
        if current_normal:
            segments.append("\n".join(current_normal).strip())
            current_normal.clear()

    def _flush_table() -> None:
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


def _infer_doc_metadata(filename: str, text: str) -> dict:
    """Infer document-level metadata from filename and content.

    Tags each document with document_type and detected parties so the LLM
    can reason about source isolation — never applying a clause from one
    document to answer a question about a different document or party.
    """
    name_lower = filename.lower()
    text_lower = text.lower()

    # Document type
    if "nda" in name_lower or "non-disclosure" in name_lower:
        doc_type = "NDA"
    elif "engagement" in name_lower or "engagement letter" in text_lower[:500]:
        doc_type = "Engagement Letter"
    elif "memo" in name_lower or "memorandum" in text_lower[:500]:
        doc_type = "Legal Memo"
    elif "contract" in name_lower or "agreement" in name_lower:
        doc_type = "Contract"
    elif "invoice" in name_lower:
        doc_type = "Invoice"
    elif "proposal" in name_lower:
        doc_type = "Proposal"
    else:
        doc_type = "Document"

    # Party detection — look for quoted defined terms like "Client" or "Counterparty"
    parties: List[str] = []
    for match in re.finditer(
        r'"([^"]{3,60}?)"\s*\(\s*"(?:Client|Company|Party|Counterparty|Employer|Employee|Disclosing Party|Receiving Party)"\s*\)',
        text,
    ):
        name = match.group(1).strip()
        if name not in parties:
            parties.append(name)

    # Also capture "between X and Y" patterns for named parties
    for match in re.finditer(
        r'\bbetween\s+([A-Z][A-Za-z &,\.]+?(?:LLC|LLP|Inc|Corp|Ltd))\s+and\s+([A-Z][A-Za-z &,\.]+?(?:LLC|LLP|Inc|Corp|Ltd))',
        text,
    ):
        for group in match.groups():
            g = group.strip()
            if g and g not in parties:
                parties.append(g)

    return {"document_type": doc_type, "parties": parties[:4]}


def _chunk_text(text: str, filename: str = "", doc_metadata: dict | None = None) -> List[str]:
    """Split text into retrieval-optimised chunks.

    Each chunk is prefixed with a rich document label that includes:
      - filename (for citation chips)
      - document_type (NDA, Engagement Letter, Legal Memo, etc.)
      - parties (named parties to the document)

    This metadata in the chunk label lets the LLM enforce source isolation —
    knowing that a clause belongs to an NDA between specific parties, not
    to an engagement letter between different parties.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    segments = _split_preserving_tables(text)

    raw_chunks: List[str] = []
    for segment in segments:
        if "|" in segment and "\n" in segment and len(segment) <= 2000:
            raw_chunks.append(segment)
        else:
            raw_chunks.extend(splitter.split_text(segment))

    doc_meta = doc_metadata or {}
    doc_type = doc_meta.get("document_type", "")
    parties = doc_meta.get("parties", [])

    # Build a rich label so the LLM always knows exactly which document and
    # which parties a chunk belongs to.
    label_parts = [f"[Document: {filename}]" if filename else ""]
    if doc_type:
        label_parts.append(f"[Type: {doc_type}]")
    if parties:
        label_parts.append(f"[Parties: {', '.join(parties)}]")
    doc_label = " ".join(p for p in label_parts if p) + "\n\n" if any(label_parts) else ""

    return [doc_label + chunk for chunk in raw_chunks if chunk.strip()]


# ── MIME type map and routing ─────────────────────────────────────────────────

_MIME_MAP = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pdf":  "application/pdf",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".txt":  "text/plain",
    ".md":   "text/markdown",
}

_SUPPORTED_EXTENSIONS = set(_MIME_MAP.keys())


def _extract_text(ext: str, file_bytes: bytes) -> str:
    """Route to the correct extractor based on file extension."""
    if ext == ".docx":
        return _extract_text_docx(file_bytes)
    elif ext == ".pdf":
        return _extract_text_pdf(file_bytes)
    elif ext == ".xlsx":
        return _extract_text_xlsx(file_bytes)
    elif ext == ".pptx":
        return _extract_text_pptx(file_bytes)
    elif ext in (".txt", ".md"):
        return _extract_text_plain(file_bytes)
    else:
        raise ValueError(
            f"Unsupported file type: '{ext}'. "
            f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
        )


# ── Embedding ─────────────────────────────────────────────────────────────────

async def _embed_texts(texts: List[str]) -> List[List[float]]:
    response = await _openai.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL,
    )
    return [item.embedding for item in response.data]


# ── Main pipeline: URL-based ingest ──────────────────────────────────────────

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
    text = _extract_text(ext, raw_bytes)

    if not text.strip():
        raise ValueError(f"Document {document_id} contains no extractable text.")

    chunks = _chunk_text(text, filename=filename, doc_metadata=_infer_doc_metadata(filename, text))
    logger.info("Document %s split into %d chunks", document_id, len(chunks))

    embeddings = await _embed_texts(chunks)

    rows = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        meta = _infer_doc_metadata(filename, text)
        row = {
            "document_id": document_id,
            "content": chunk,
            "embedding": embedding,
            "chunk_index": idx,
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                "document_type": meta.get("document_type", ""),
                "parties": meta.get("parties", []),
            },
        }
        if user_id:
            row["user_id"] = user_id
        rows.append(row)

    supabase.table("chunks").insert(rows).execute()
    supabase.table("documents").update({"status": "completed"}).eq("id", document_id).execute()

    logger.info("Document %s ingested successfully (%d chunks)", document_id, len(chunks))
    return len(chunks)


# ── Main pipeline: bytes-based ingest (upload endpoint) ──────────────────────

async def ingest_document_from_bytes(
    document_id: str,
    filename: str,
    file_bytes: bytes,
    user_id: str | None = None,
) -> int:
    """Ingest from raw file bytes (the /upload endpoint). Returns chunks stored."""
    ext = os.path.splitext(filename)[1].lower()

    text = _extract_text(ext, file_bytes)

    if not text.strip():
        raise ValueError(f"Document '{filename}' contains no extractable text.")

    doc_row: dict = {
        "id": document_id,
        "title": filename,
        "status": "processing",
        "mime_type": _MIME_MAP.get(ext, "application/octet-stream"),
        "file_size": len(file_bytes),
    }
    if user_id:
        doc_row["user_id"] = user_id

    try:
        supabase.table("documents").upsert(doc_row).execute()
    except Exception as e:
        logger.warning("Could not upsert document row for %s: %s – continuing", document_id, e)

    doc_meta = _infer_doc_metadata(filename, text)
    chunks = _chunk_text(text, filename=filename, doc_metadata=doc_meta)
    logger.info("Upload '%s' (%s) split into %d chunks", filename, document_id, len(chunks))

    embeddings = await _embed_texts(chunks)

    rows = []
    for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        row = {
            "document_id": document_id,
            "content": chunk,
            "embedding": embedding,
            "chunk_index": idx,
            "metadata": {
                "filename": filename,
                "chunk_index": idx,
                "document_type": doc_meta.get("document_type", ""),
                "parties": doc_meta.get("parties", []),
            },
        }
        if user_id:
            row["user_id"] = user_id
        rows.append(row)

    supabase.table("chunks").insert(rows).execute()

    try:
        supabase.table("documents").update({"status": "completed"}).eq("id", document_id).execute()
    except Exception:
        logger.warning("Could not update document status for %s", document_id)

    logger.info("Upload '%s' ingested successfully (%d chunks)", filename, document_id)
    return len(chunks)
