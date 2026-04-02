"""CorporateBrain – RAG query service.

Pipeline: embed question → optional multi-query expansion → vector search
         → deduplicate → optional rerank → build prompt → LLM answer.

Key features:
  - Aria identity: consistent AI persona across all interactions.
  - Session memory: recent conversation turns are injected into the prompt
    so Aria can reference prior exchanges within a session.
  - Similarity threshold lowered from 0.5 → 0.3 for better recall.
  - Default top_k raised to 7 for better coverage.
  - Multi-query expansion for synthesis questions.
  - LLM-based reranking: chunks are re-scored for relevance before answer generation.
  - Source citations show the actual document filename.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Optional

from openai import AsyncOpenAI

from services.db import supabase

logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Similarity threshold — lowered from 0.5 to catch more relevant chunks.
SIMILARITY_THRESHOLD = 0.3

# In-memory session store: session_id → [{"role": ..., "content": ...}, ...]
# Sessions are capped to MAX_HISTORY turns per session.
_session_history: Dict[str, List[dict]] = {}
MAX_HISTORY_TURNS = 6   # 3 user + 3 assistant messages

GENERAL_CHAT_KEYWORDS = {
    "hello", "hi", "hey", "help", "what can you do", "how does", "summarize",
    "pipeline", "lead", "leads", "system", "vera", "chat", "email", "outreach",
}

RAG_INTENT_KEYWORDS = {
    "document", "documents", "knowledge base", "kb", "policy", "playbook", "pdf",
    "file", "files", "uploaded", "upload", "source", "sources", "pricing", "retainer",
    "case study", "proposal", "contract", "scope", "timeline", "reference", "cite",
}

BUSINESS_QUERY_KEYWORDS = {
    "agency", "client", "clients", "lead", "leads", "pricing", "retainer", "service",
    "services", "proposal", "project", "pipeline", "qualification", "outreach", "email",
    "sales", "conversion", "close", "playbook", "onboarding", "timeline",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _embed_query(text: str) -> List[float]:
    response = await _openai.embeddings.create(input=[text], model=EMBEDDING_MODEL)
    return response.data[0].embedding


def _vector_search(
    embedding: List[float],
    user_id: Optional[str] = None,
    top_k: int = 7,
    threshold: float = SIMILARITY_THRESHOLD,
) -> List[dict]:
    """Call the Supabase match_knowledge RPC."""
    params: dict = {
        "query_embedding": embedding,
        "match_threshold": threshold,
        "match_count": top_k,
    }
    if user_id:
        params["filter_user_id"] = user_id

    result = supabase.rpc("match_knowledge", params).execute()
    return result.data or []


async def _expand_query(question: str) -> List[str]:
    """Generate 2 alternative phrasings of the question for better recall."""
    try:
        completion = await _openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You rephrase questions to improve document retrieval. "
                        "Given a question, output exactly 2 alternative phrasings "
                        "using different vocabulary but asking the same thing. "
                        "Output only the 2 phrasings, one per line, no numbering."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        raw = completion.choices[0].message.content or ""
        extras = [l.strip() for l in raw.strip().splitlines() if l.strip()]
        return [question] + extras[:2]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        return [question]


def _deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    """Remove duplicates (same chunk id), keeping highest similarity."""
    seen: dict[str, dict] = {}
    for chunk in chunks:
        cid = str(chunk.get("id", ""))
        if cid not in seen or chunk.get("similarity", 0) > seen[cid].get("similarity", 0):
            seen[cid] = chunk
    return sorted(seen.values(), key=lambda c: c.get("similarity", 0), reverse=True)


async def _rerank_chunks(
    question: str,
    chunks: List[dict],
    top_k: int,
) -> List[dict]:
    """LLM-based reranking: score each chunk 0-10 for relevance, keep top_k.

    Falls back to similarity ordering on any error.
    """
    if len(chunks) <= top_k:
        return chunks

    numbered = "\n\n".join(
        f"[{i+1}] {chunk.get('content', '')[:400]}"
        for i, chunk in enumerate(chunks)
    )
    try:
        completion = await _openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a relevance judge. Given a question and numbered text "
                        "chunks, output a JSON array of objects with 'index' (1-based) "
                        "and 'score' (0-10). Score 10 = directly answers the question, "
                        "0 = completely irrelevant. Output ONLY the JSON array."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\nChunks:\n{numbered}"
                    ),
                },
            ],
        )
        import json
        raw = (completion.choices[0].message.content or "").strip()
        # Strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        scores: List[dict] = json.loads(raw)
        score_map = {int(s["index"]) - 1: float(s["score"]) for s in scores}

        # Sort by rerank score descending, break ties by similarity
        ranked = sorted(
            enumerate(chunks),
            key=lambda t: (score_map.get(t[0], 0), t[1].get("similarity", 0)),
            reverse=True,
        )
        return [chunk for _, chunk in ranked[:top_k]]

    except Exception as exc:
        logger.warning("Reranking failed, using similarity order: %s", exc)
        return chunks[:top_k]


# ── Session memory helpers ────────────────────────────────────────────────────

def _get_history(session_id: str) -> List[dict]:
    return _session_history.get(session_id, [])


def _push_history(session_id: str, role: str, content: str) -> None:
    history = _session_history.setdefault(session_id, [])
    history.append({"role": role, "content": content})
    # Keep only the last MAX_HISTORY_TURNS messages
    if len(history) > MAX_HISTORY_TURNS * 2:
        _session_history[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


def _looks_general_chat(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return True

    if any(k in q for k in BUSINESS_QUERY_KEYWORDS):
        return False

    if any(k in q for k in RAG_INTENT_KEYWORDS):
        return False

    if len(q.split()) <= 3:
        return True

    # Natural, broad assistant asks should stay conversational.
    if q.startswith(("what can", "how do", "can you", "help me", "explain")):
        return True

    return any(k in q for k in GENERAL_CHAT_KEYWORDS)


async def _answer_general_chat(question: str, session_id: Optional[str] = None) -> str:
    """Respond as a helpful assistant when retrieval is not needed."""
    messages: List[dict] = [
        {
            "role": "system",
            "content": (
                "You are Aria, an AI employee on this agency's team. "
                "If asked your name or identity, answer in first person singular: 'I am Aria'. "
                "Never say 'we go by Aria'. "
                "When discussing business operations, speak in first-person team language "
                "like 'we', 'our', and 'the team'. "
                "Respond clearly and conversationally. "
                "Do not mention missing database knowledge unless explicitly asked "
                "for a factual detail from uploaded documents."
            ),
        }
    ]
    if session_id:
        messages.extend(_get_history(session_id))
    messages.append({"role": "user", "content": question})

    completion = await _openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.4,
        messages=messages,
    )
    return completion.choices[0].message.content or ""


# ── System prompt – Aria identity ─────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are **Aria**, an AI employee and digital brain of this agency. \
You are knowledgeable, professional, and slightly personable — like a highly capable senior \
analyst who genuinely cares about getting the right answer.

You are answering using company knowledge context provided below. \
Use that context as your source of truth for factual claims.

When answering business-related questions, respond as part of the internal team using "we" and "our".
If asked your name or identity, respond in first person singular with: "I am Aria".

Core rules:
1. Answer directly and concisely. No filler phrases like "Great question!" or "Certainly!".
2. If the context contains the answer — even phrased differently — extract and deliver it.
3. Numbers, prices, metrics, and dates in the context are facts — cite them precisely.
4. Synthesise across multiple documents when the answer spans sources.
5. Only say you cannot find information when it is genuinely absent from ALL provided context.
6. Do not add a "Sources" section or bracket citations in the answer body.
7. If a previous conversation turn is relevant to this question, reference it naturally \
   (e.g., "As I mentioned earlier…").
8. Keep responses structured — use bullet points or numbered lists where they improve clarity.
"""


def _strip_document_label(text: str) -> str:
    """Remove leading [Document: ...] labels from chunk content."""
    return re.sub(r"^\s*\[Document:[^\]]+\]\s*\n*", "", text).strip()


def _clean_answer_output(text: str) -> str:
    """Remove leaked retrieval labels from answer text."""
    cleaned = text or ""
    cleaned = re.sub(r"\[Document:[^\]]+\]", "", cleaned)
    cleaned = re.sub(r"\n\s*Sources?:[\s\S]*$", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


# ── Main RAG pipeline ─────────────────────────────────────────────────────────

async def query_knowledge(
    question: str,
    user_id: Optional[str] = None,
    top_k: int = 7,
    session_id: Optional[str] = None,
) -> dict:
    """Full RAG pipeline: expand query → embed → search → dedup → rerank → answer.

    Parameters
    ----------
    question   : the user's natural-language question
    user_id    : optional Supabase user UUID for scoped search
    top_k      : number of chunks to retrieve (and rerank down to)
    session_id : if provided, conversation history is maintained per session

    Returns {"answer": str, "sources": list[dict]}.
    """
    logger.info("Query (session=%s): %.80s…", session_id, question)

    # 0. General conversational mode for non-document questions.
    # This makes chat feel intelligent even when no KB lookup is needed.
    if _looks_general_chat(question):
        answer = await _answer_general_chat(question, session_id=session_id)
        if session_id:
            _push_history(session_id, "user", question)
            _push_history(session_id, "assistant", answer)
        return {"answer": answer, "sources": []}

    # 1. Expand query
    queries = await _expand_query(question)
    logger.info("Running %d query variants", len(queries))

    # 2. Embed + search in parallel
    import asyncio
    embeddings = await asyncio.gather(*[_embed_query(q) for q in queries])

    all_chunks: List[dict] = []
    for emb in embeddings:
        results = _vector_search(emb, user_id=user_id, top_k=top_k, threshold=SIMILARITY_THRESHOLD)
        all_chunks.extend(results)

    # 3. Deduplicate
    chunks = _deduplicate_chunks(all_chunks)

    # 4. LLM rerank (fetches extra chunks then cuts down to top_k)
    chunks = await _rerank_chunks(question, chunks, top_k=top_k)

    if not chunks:
        answer = await _answer_general_chat(question, session_id=session_id)
        if session_id:
            _push_history(session_id, "user", question)
            _push_history(session_id, "assistant", answer)
        return {"answer": answer, "sources": []}

    # 5. Build context block with document labels
    context_parts = []
    sources = []
    for i, chunk in enumerate(chunks, 1):
        content = chunk["content"]
        doc_name = ""
        if content.startswith("[Document:"):
            end = content.find("]")
            if end != -1:
                doc_name = content[len("[Document:"):end].strip()

        clean_content = _strip_document_label(content)

        context_parts.append(f"[Source {i} — {doc_name or 'Unknown'}]\n{clean_content}")
        sources.append({
            "chunk_id": chunk.get("id"),
            "document_id": chunk.get("document_id"),
            "document_name": doc_name,
            "similarity": round(chunk.get("similarity", 0), 4),
            "relevance": round(chunk.get("similarity", 0) * 100),
            "snippet": clean_content[:200] + "…" if len(clean_content) > 200 else clean_content,
        })

    context_block = "\n\n---\n\n".join(context_parts)
    user_message = (
        f"Context from knowledge base:\n\n{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    # 6. Build messages list — inject session history for memory
    messages: List[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if session_id:
        history = _get_history(session_id)
        messages.extend(history)

    messages.append({"role": "user", "content": user_message})

    # 7. Generate answer
    logger.info("Generating answer from %d chunks (session=%s)", len(chunks), session_id)
    completion = await _openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.1,
        messages=messages,
    )

    answer = _clean_answer_output(completion.choices[0].message.content or "")
    logger.info("Answer: %.80s…", answer)

    # 8. Persist session memory
    if session_id:
        _push_history(session_id, "user", question)
        _push_history(session_id, "assistant", answer)

    return {"answer": answer, "sources": sources}


