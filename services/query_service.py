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

import json
import logging
import os
import re
import asyncio
import time
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi
from openai import AsyncOpenAI

from services.db import supabase
from services.persona import get_persona_config

logger = logging.getLogger(__name__)

# ── Client ────────────────────────────────────────────────────────────────────
_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)

EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Baseline threshold; runtime is adapted by query style in _dynamic_threshold.
SIMILARITY_THRESHOLD = 0.3

MIN_THRESHOLD = 0.18
MAX_THRESHOLD = 0.45
RETRIEVAL_CANDIDATE_MULTIPLIER = 4
RERANK_SNIPPET_CHARS = 900
DEFAULT_MAX_QUERY_VARIANTS = 4
DIRECT_PROBE_CONFIDENCE = 0.50
EXPANSION_CONFIDENCE_FLOOR = 0.32
RERANK_MIN_CANDIDATES = 6

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
    "firm", "matter", "case", "billing", "conflict", "deadline", "court", "hearing",
}

CASUAL_QUERY_MARKERS = {
    "kinda", "sorta", "idk", "lol", "pls", "thx", "u", "ya", "btw",
    "any idea", "can u", "gimme", "wanna", "stuff", "things", "whatever",
}

FAST_OUTREACH_MARKERS = {
    "draft email", "write email", "first-touch email", "outreach email",
    "cold email", "follow-up email", "intro email",
}

# ── Persona-specific domain synonym maps ──────────────────────────────────────
# Agency reference (kept for future demos):
# DOMAIN_SYNONYM_MAP originally only covered agency acronyms and deliverables.
_AGENCY_DOMAIN_SYNONYM_MAP = {
    "monthly package": ["retainer", "monthly retainer", "ongoing package"],
    "price": ["pricing", "rate", "cost", "fee"],
    "website": ["web project", "site build", "web design"],
    "brand": ["branding", "brand strategy", "identity"],
    "timeline": ["delivery timeline", "turnaround", "schedule"],
    "proposal": ["scope", "statement of work", "quote"],
    # Agency performance acronyms — query expander won't catch these unprompted
    "roas": ["return on ad spend", "revenue per ad dollar", "ad return", "return on advertising spend"],
    "cpl": ["cost per lead", "lead cost", "acquisition cost per lead", "cost to acquire lead"],
    "ctr": ["click-through rate", "click rate", "clicks per impression"],
    "cac": ["customer acquisition cost", "cost to acquire", "cost per acquisition"],
    "cpa": ["cost per acquisition", "cost per action", "conversion cost"],
    "ltv": ["lifetime value", "customer lifetime value", "long-term value"],
    "mrr": ["monthly recurring revenue", "monthly revenue", "recurring revenue"],
    "kpi": ["key performance indicator", "performance metric", "success metric"],
    "icp": ["ideal customer profile", "target customer", "buyer persona"],
    # Document type synonyms for agency deliverables
    "brief": ["creative brief", "campaign brief", "project brief", "scope"],
    "deck": ["presentation", "slide deck", "pitch deck", "slides"],
    "report": ["performance report", "monthly report", "campaign results", "analytics report"],
    "sop": ["standard operating procedure", "process document", "playbook", "guidelines"],
    "case study": ["client success story", "result showcase", "client outcome"],
}

_LAW_FIRM_DOMAIN_SYNONYM_MAP = {
    "practice area": ["legal practice", "service line", "matter category"],
    "intake": ["matter intake", "client intake", "new matter screening"],
    "billing": ["fee structure", "hourly rate", "flat fee", "retainer"],
    "retainer": ["engagement retainer", "advance fee", "trust deposit"],
    "conflict": ["conflict check", "conflicts review", "conflict screening"],
    "deadline": ["filing deadline", "court date", "limitation period"],
    "settlement": ["settlement value", "settlement range", "settlement posture"],
    "litigation": ["dispute", "lawsuit", "court proceeding"],
    "contract": ["agreement", "commercial contract", "contract clause"],
    "compliance": ["regulatory", "policy compliance", "legal requirement"],
    "precedent": ["case law", "prior decision", "authority"],
    "matter": ["case", "file", "engagement"],
}


def _domain_synonym_map() -> Dict[str, List[str]]:
    persona = get_persona_config()
    if persona.key == "agency":
        return _AGENCY_DOMAIN_SYNONYM_MAP
    return _LAW_FIRM_DOMAIN_SYNONYM_MAP


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
    """Generate diverse rewrites for better recall on casual/unusual queries.

    Returns up to 5 variants including:
      - original query
      - normalized professional rewrite
      - keyword-focused retrieval query
      - acronym/synonym-expanded variant
      - typo-corrected variant
    """
    try:
        completion = await _openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You optimize messy user queries for enterprise retrieval. "
                        "Return STRICT JSON with exactly these keys: "
                        "professional, keywords, expanded, typo_fixed. "
                        "Rules: preserve intent, avoid adding facts, keep each under 22 words. "
                        "'keywords' should be compact noun phrases likely to appear in docs."
                    ),
                },
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
        )
        raw = completion.choices[0].message.content or "{}"
        data = json.loads(raw)
        variants = [
            question,
            str(data.get("professional", "")).strip(),
            str(data.get("keywords", "")).strip(),
            str(data.get("expanded", "")).strip(),
            str(data.get("typo_fixed", "")).strip(),
        ]
        # De-duplicate while preserving order.
        seen = set()
        deduped = []
        for v in variants:
            if not v:
                continue
            key = v.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(v)
        return deduped[:5]
    except Exception as exc:
        logger.warning("Query expansion failed: %s", exc)
        # Deterministic fallback rewrite path.
        normalized = re.sub(r"\s+", " ", question).strip()
        keywordish = " ".join([w for w in re.findall(r"[a-zA-Z0-9]+", normalized.lower()) if len(w) > 2])
        return [normalized, keywordish][:2] if keywordish else [normalized]


def _tokenize(text: str) -> List[str]:
    tokens = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return [t for t in tokens if len(t) > 2]


def _strip_noise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _apply_domain_synonyms(question: str) -> str:
    q = question
    lower = q.lower()
    for phrase, expansions in _domain_synonym_map().items():
        if phrase in lower:
            q += " " + " ".join(expansions)
    return _strip_noise(q)


def _decompose_query(question: str) -> List[str]:
    """Split compound asks into atomic sub-questions for multi-hop retrieval."""
    q = _strip_noise(question)
    if len(q.split()) < 9:
        return [q]

    parts = re.split(r"\b(?:and|then|also|plus)\b|[,;]", q, flags=re.IGNORECASE)
    parts = [p.strip(" .") for p in parts if p and p.strip()]

    if len(parts) <= 1:
        return [q]

    # Keep the full query plus up to 3 atomic chunks.
    return [q] + parts[:3]


def _dynamic_threshold(question: str, *, probe: bool = False) -> float:
    """Adapt similarity threshold to query shape.

    Casual, short, or vague queries get a lower threshold for recall.
    Specific longer queries get a stricter threshold for precision.
    """
    q = question.lower().strip()
    wc = len(q.split())

    threshold = SIMILARITY_THRESHOLD
    if wc <= 4:
        threshold -= 0.08
    elif wc >= 14:
        threshold += 0.05

    if any(marker in q for marker in CASUAL_QUERY_MARKERS):
        threshold -= 0.06

    if any(k in q for k in ("exact", "specifically", "number", "price", "rate", "timeline")):
        threshold += 0.04

    if probe:
        threshold -= 0.03

    return max(MIN_THRESHOLD, min(MAX_THRESHOLD, threshold))


def _lexical_fuzzy_score(question: str, chunk_content: str) -> float:
    """BM25-based lexical relevance score for a single chunk against the query.

    BM25 weights term frequency and document length, making it significantly
    more accurate than SequenceMatcher for keyword-overlap scoring.
    We treat each chunk as a single-document corpus so BM25 degrades
    gracefully to a TF-IDF-like score.
    """
    q_tokens = _tokenize(question)
    if not q_tokens:
        return 0.0

    text = _strip_document_label(chunk_content)
    c_tokens = _tokenize(text)
    if not c_tokens:
        return 0.0

    bm25 = BM25Okapi([c_tokens])
    scores = bm25.get_scores(q_tokens)
    raw = float(scores[0])
    # Normalise: BM25 scores are unbounded; cap at 20 and scale to [0, 1].
    return min(raw / 20.0, 1.0)


def _hybrid_fuse(question: str, chunks: List[dict]) -> List[dict]:
    """Fuse dense similarity with lexical+fuzzy signals using weighted RRF."""
    if not chunks:
        return []

    dense_sorted = sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True)
    lexical_sorted = sorted(
        chunks,
        key=lambda c: _lexical_fuzzy_score(question, c.get("content", "")),
        reverse=True,
    )

    dense_rank = {str(c.get("id")): i for i, c in enumerate(dense_sorted)}
    lexical_rank = {str(c.get("id")): i for i, c in enumerate(lexical_sorted)}

    fused: List[Tuple[float, dict]] = []
    for c in chunks:
        cid = str(c.get("id"))
        d = dense_rank.get(cid, 999)
        l = lexical_rank.get(cid, 999)
        # Weighted reciprocal rank fusion.
        score = (0.65 / (d + 1)) + (0.35 / (l + 1))
        c2 = dict(c)
        c2["hybrid_score"] = score
        fused.append((score, c2))

    fused.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in fused]


def _retrieval_confidence(chunks: List[dict]) -> float:
    if not chunks:
        return 0.0
    top = float(chunks[0].get("similarity", 0) or 0)
    top3 = chunks[:3]
    avg = sum(float(c.get("similarity", 0) or 0) for c in top3) / max(1, len(top3))
    gap = top - float(chunks[1].get("similarity", 0) or 0) if len(chunks) > 1 else top
    # Conservative bounded blend.
    conf = (0.55 * top) + (0.35 * avg) + (0.10 * max(0.0, gap))
    return max(0.0, min(1.0, conf))


def _needs_clarification(question: str, chunks: List[dict], confidence: float) -> bool:
    # Threshold lowered to 0.22: agency docs score lower cosine similarity
    # against conversational queries so 0.33 was firing too aggressively.
    # Word-count trigger removed: a long specific question with chunks found
    # should never be redirected to clarification.
    if confidence >= 0.22:
        return False
    q = question.lower()
    likely_factual = any(k in q for k in ("price", "pricing", "rate", "timeline", "cost", "what", "how much", "when"))
    # Only clarify if the question is genuinely vague: short AND low confidence.
    return likely_factual and len(q.split()) <= 4


def _is_fast_outreach_request(question: str) -> bool:
    q = (question or "").lower()
    if not q:
        return False
    has_email_intent = any(m in q for m in FAST_OUTREACH_MARKERS)
    has_generation_verb = any(v in q for v in ("write", "draft", "generate", "compose"))
    return has_email_intent or ("email" in q and has_generation_verb)


def _should_expand_query(question: str, probe_conf: float) -> bool:
    """Gate expensive LLM query expansion behind low-confidence retrieval signals."""
    q = (question or "").lower().strip()
    if not q:
        return False
    if probe_conf < EXPANSION_CONFIDENCE_FLOOR:
        return True
    if len(q.split()) <= 4:
        return True
    return any(marker in q for marker in CASUAL_QUERY_MARKERS)


def _should_rerank(chunks: List[dict], confidence: float) -> bool:
    """Only use LLM reranking when candidate ambiguity is likely."""
    if len(chunks) < RERANK_MIN_CANDIDATES:
        return False
    return confidence < 0.56


async def _clarify_question(question: str) -> str:
    """Produce a concise clarification question while staying conversational."""
    persona = get_persona_config()
    try:
        completion = await _openai.chat.completions.create(
            model=CHAT_MODEL,
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are {persona.assistant_name}. Ask one concise clarification question to disambiguate "
                        "the user's request for document-backed lookup. "
                        "Offer 2-3 short options. Keep under 35 words."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        text = (completion.choices[0].message.content or "").strip()
        if text:
            return text
    except Exception as exc:
        logger.warning("Clarification generation failed: %s", exc)

    persona = get_persona_config()
    if persona.key == "agency":
        return (
            "I can help, but I need a bit more direction. Do you mean pricing, timeline, "
            "service scope, or a specific campaign or client context?"
        )
    return (
        "I can help, but I need a bit more direction. Do you mean billing, intake workflow, "
        "a specific matter, or a practice-area question?"
    )


def _deduplicate_chunks(chunks: List[dict]) -> List[dict]:
    """Remove duplicates (same chunk id), keeping highest similarity."""
    seen: dict[str, dict] = {}
    for chunk in chunks:
        cid = str(chunk.get("id", ""))
        if cid not in seen or chunk.get("similarity", 0) > seen[cid].get("similarity", 0):
            seen[cid] = chunk
    return sorted(seen.values(), key=lambda c: c.get("similarity", 0), reverse=True)


def _extract_parties_from_question(question: str) -> List[str]:
    """Extract named company/person entities from the question.

    Looks for:
      - Known legal entity suffixes (LLC, LLP, Inc, Corp, Ltd)
      - Title-cased multi-word names that appear before relational verbs

    Returns a list of normalised lowercase name fragments so matching is
    case-insensitive and partial (e.g. "luminary" matches "Luminary Ventures Inc.").
    """
    parties: List[str] = []

    # Pattern 1: formal entity names with suffixes
    for match in re.finditer(
        r'\b([A-Z][A-Za-z &,\.]+?(?:LLC|LLP|Inc|Corp|Ltd|LP))\b',
        question,
    ):
        name = match.group(1).strip(" ,.")
        token = name.lower()
        if token not in parties:
            parties.append(token)

    # Pattern 2: Title Case runs of 2+ words (names without entity suffix)
    # e.g. "Bloom Health", "Apex Insurance"
    for match in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', question):
        name = match.group(1).strip()
        token = name.lower()
        if not any(token in p for p in parties) and token not in parties:
            parties.append(token)

    return parties


def _scope_chunks_by_party(question: str, chunks: List[dict]) -> List[dict]:
    """Filter and boost chunks based on party match with the question.

    When a question names specific parties (e.g. "Bloom Health" or "Apex Insurance"),
    chunks are handled as follows:

      HARD EXCLUDE: chunk has explicit [Parties: ...] metadata and NONE of
        those parties match any party named in the question. These chunks are
        from the wrong document entirely and must not reach the LLM.

      BOOST (+0.15 per match, capped at +0.30): chunk parties overlap with
        question parties — this is the right document.

      NO CHANGE: chunk has no [Parties: ...] label (unlabelled docs are kept
        as candidates since most agency docs don't have party metadata).
    """
    question_parties = _extract_parties_from_question(question)

    if not question_parties:
        return chunks

    scoped = []
    excluded = 0

    for c in chunks:
        content = c.get("content", "")
        meta = c.get("metadata") or {}
        chunk_parties: List[str] = meta.get("parties", [])

        # Also try to extract party tags from the chunk label text itself
        label_match = re.search(r'\[Parties:\s*([^\]]+)\]', content)
        if label_match and not chunk_parties:
            chunk_parties = [p.strip().lower() for p in label_match.group(1).split(",")]

        if chunk_parties:
            matches = sum(
                1 for qp in question_parties
                if any(qp in cp or cp in qp for cp in chunk_parties)
            )
            if matches == 0:
                excluded += 1
                continue
            boost = min(0.30, matches * 0.15)
            c2 = dict(c)
            c2["similarity"] = min(1.0, float(c.get("similarity", 0)) + boost)
            c2["_party_adjustment"] = boost
            scoped.append(c2)
        else:
            c["_party_adjustment"] = 0.0
            scoped.append(c)

    if excluded:
        logger.info(
            "Party scoping excluded %d chunks (question parties: %s)",
            excluded, question_parties,
        )

    # Re-sort after adjustments
    scoped.sort(key=lambda c: c.get("similarity", 0), reverse=True)

    # Safety: if hard exclusion removed everything, fall back to full set
    # (this prevents a silent empty-context hallucination)
    if not scoped:
        logger.warning("Party scoping excluded all chunks — falling back to unscoped results")
        return sorted(chunks, key=lambda c: c.get("similarity", 0), reverse=True)

    return scoped


async def _rerank_chunks(
    question: str,
    chunks: List[dict],
    top_k: int,
) -> List[dict]:
    """LLM-based reranking: score each chunk 0-10 for relevance, keep top_k.

    Falls back to similarity ordering on any error.
    Always runs — even when chunk count <= top_k — so ordering reflects true
    relevance rather than raw cosine similarity.
    """
    if not chunks:
        return chunks

    numbered = "\n\n".join(
        f"[{i+1}] {chunk.get('content', '')[:RERANK_SNIPPET_CHARS]}"
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
    persona = get_persona_config()
    messages: List[dict] = [
        {
            "role": "system",
            "content": (
                f"You are {persona.assistant_name}, an AI employee on this {persona.org_noun}'s team. "
                f"If asked your name or identity, answer in first person singular: 'I am {persona.assistant_name}'. "
                f"Never say 'we go by {persona.assistant_name}'. "
                "Assume the user is an internal colleague, not an external customer. "
                "Avoid customer-support phrasing like apologies or service-agent language. "
                f"You work inside Vera, an AI operating system for {persona.org_noun}s with three core jobs: "
                "(1) knowledge retrieval from uploaded docs, "
                "(2) lead analysis and pipeline support, and "
                "(3) outreach drafting. "
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


def _build_system_prompt() -> str:
    persona = get_persona_config()

    # Agency reference (kept for future demos):
    # You are **Aria**, an AI employee and digital brain of this agency.
    if persona.key == "agency":
        return f"""\
You are **{persona.assistant_name}**, an AI employee and digital brain of this agency. \
You are knowledgeable, professional, and slightly personable — like a highly capable senior \
analyst who genuinely cares about getting the right answer.

You are answering using company knowledge context provided below. \
Use that context as your source of truth for factual claims.

If asked your name or identity, respond in first person singular with: "I am {persona.assistant_name}".
Treat the user as an internal colleague by default, not an external customer.
Do not use customer-support language or formal support disclaimers.
You operate within Vera, which supports knowledge Q&A, lead intelligence, and outreach drafting.

Core rules:
1. Answer directly and concisely. No filler phrases like "Great question!" or "Certainly!".
2. If the context contains the answer — even phrased differently — extract and deliver it.
3. Numbers, prices, metrics, and dates in the context are facts — cite them precisely.
4. Synthesise across multiple documents when the answer spans sources.
4b. If two source documents give DIFFERENT answers to the same question (e.g. different \
    retainer rates, different onboarding timelines), explicitly state both answers and identify \
    which document each comes from. Never silently pick one — surface the conflict.
5. Only say you cannot find information when it is genuinely absent from ALL provided context.
6. Do not add a "Sources" section or bracket citations in the answer body.
7. If a previous conversation turn is relevant to this question, reference it naturally \
   (e.g., "As I mentioned earlier...").
8. Keep responses structured — use bullet points or numbered lists where they improve clarity.
9. SOURCE ISOLATION — CRITICAL: Every clause, threshold, obligation, or term belongs to a \
   specific document and a specific set of parties or clients. NEVER apply information from \
   Document A to answer a question about Document B. If a question names a specific client, \
   campaign, or brief, only use chunks explicitly sourced from that document. If you find \
   relevant-sounding content but it comes from a different client or document, say so \
   explicitly rather than applying it across contexts.
10. MATH AND THRESHOLD REASONING: If a question involves a quantity and a threshold or rate \
    from the documents, always compute the arithmetic explicitly before concluding. \
    Show the calculation step before stating the answer — never skip to a conclusion. \
    Examples: "$4,200 ad spend × 4.5x ROAS = $18,900 attributed revenue". \
    Cite exact figures from the document — do not round or approximate.
"""

    return f"""\
You are **{persona.assistant_name}**, an AI employee and legal knowledge operator for this law firm. \
You are knowledgeable, precise, and practical — like a strong legal operations lead.

You are answering using firm knowledge context provided below. \
Use that context as your source of truth for factual claims.

If asked your name or identity, respond in first person singular with: "I am {persona.assistant_name}".
Treat the user as an internal colleague by default, not an external client.
Do not use customer-support language or formal support disclaimers.
You operate within Vera, which supports knowledge Q&A, lead intelligence, and outreach drafting.

Core rules:
1. Answer directly and concisely. No filler phrases like "Great question".
2. If context contains the answer, extract and deliver it clearly.
3. Numbers, deadlines, rates, and dates in context are facts — quote them precisely.
4. Synthesize across documents when needed, but keep source boundaries strict.
5. If documents conflict (e.g., different fee ranges or timelines), state both and identify each source.
6. Only say information is missing if it is genuinely absent from all provided context.
7. Do not add a separate Sources section or bracket citations in the answer body.
8. Use structured output (bullets or numbered steps) where it improves clarity.
9. SOURCE ISOLATION — CRITICAL: Every clause, obligation, or threshold belongs to a specific \
   document and set of parties. Never apply terms from one matter to another.
10. If a question involves arithmetic (fees, hours, totals, deadlines), show the calculation step \
    before concluding and use exact document figures.
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
    start_ts = time.perf_counter()
    logger.info("Query (session=%s): %.80s…", session_id, question)

    # 0. Dual-pass routing: probe retrieval first, then decide path.
    # This avoids brittle intent routing for casual/unusual phrasings.
    probe_threshold = _dynamic_threshold(question, probe=True)
    try:
        probe_emb = await _embed_query(question)
        probe_chunks = await asyncio.to_thread(
            _vector_search,
            probe_emb,
            user_id,
            max(4, min(8, top_k)),
            probe_threshold,
        )
        probe_chunks = _deduplicate_chunks(probe_chunks)
    except Exception as exc:
        logger.warning("Probe retrieval failed: %s", exc)
        probe_chunks = []

    probe_conf = _retrieval_confidence(probe_chunks)
    prefer_general = _looks_general_chat(question)
    should_use_rag = (probe_conf >= 0.18) or not prefer_general

    if not should_use_rag:
        answer = await _answer_general_chat(question, session_id=session_id)
        if session_id:
            _push_history(session_id, "user", question)
            _push_history(session_id, "assistant", answer)
        logger.info("Query completed via general chat path in %.2fms", (time.perf_counter() - start_ts) * 1000)
        return {"answer": answer, "sources": []}

    if probe_conf >= DIRECT_PROBE_CONFIDENCE and len(probe_chunks) >= top_k:
        logger.info("Using direct probe fast path (conf=%.3f, chunks=%d)", probe_conf, len(probe_chunks))
        chunks = probe_chunks[:top_k]
        confidence = _retrieval_confidence(chunks)
    else:
        try:
            # 1-4. Retrieval strategy: fast path for outreach drafting, full path otherwise.
            fast_outreach = _is_fast_outreach_request(question)
            if fast_outreach:
                logger.info("Using fast outreach retrieval path")
                query_text = _apply_domain_synonyms(question)
                emb = await _embed_query(query_text)
                search_top_k = max(min(top_k + 2, 8), top_k)
                chunks = await asyncio.to_thread(
                    _vector_search,
                    emb,
                    user_id,
                    search_top_k,
                    _dynamic_threshold(question, probe=True),
                )
                chunks = _deduplicate_chunks(chunks)[:top_k]
            else:
                # 1. Expand query only when probe confidence is low.
                queries: List[str] = [question]
                if _should_expand_query(question, probe_conf):
                    queries = await _expand_query(question)
                # Add domain-aware synonym expansion and multi-hop decomposition.
                queries.extend(_decompose_query(question))
                queries.append(_apply_domain_synonyms(question))
                # De-duplicate query list.
                dedup_queries: List[str] = []
                seen_queries = set()
                for q in queries:
                    k = q.lower().strip()
                    if not k or k in seen_queries:
                        continue
                    seen_queries.add(k)
                    dedup_queries.append(q)
                queries = dedup_queries[:DEFAULT_MAX_QUERY_VARIANTS]
                logger.info("Running %d query variants", len(queries))

                # 2. Embed in parallel
                embeddings = await asyncio.gather(*[_embed_query(q) for q in queries])

                # 3. Run vector search calls off-event-loop in parallel.
                search_top_k = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
                search_threshold = _dynamic_threshold(question)
                search_tasks = [
                    asyncio.to_thread(
                        _vector_search,
                        emb,
                        user_id,
                        search_top_k,
                        search_threshold,
                    )
                    for emb in embeddings
                ]
                search_results = await asyncio.gather(*search_tasks)

                all_chunks: List[dict] = []
                for results in search_results:
                    all_chunks.extend(results)

                # 4. Deduplicate and apply ranking helpers.
                chunks = _deduplicate_chunks(all_chunks)

                # 4a. Party-scoped re-ranking: boost chunks from documents whose
                # parties match named entities in the question; demote others.
                chunks = _scope_chunks_by_party(question, chunks)

                # 4b. Hybrid fusion (dense + lexical/fuzzy)
                chunks = _hybrid_fuse(question, chunks)
                candidate_cap = max(top_k * RETRIEVAL_CANDIDATE_MULTIPLIER, top_k)
                chunks = chunks[:candidate_cap]

                # 4c. LLM rerank only when confidence/candidate count warrants it.
                pre_rerank_conf = _retrieval_confidence(chunks)
                if _should_rerank(chunks, pre_rerank_conf):
                    chunks = await _rerank_chunks(question, chunks, top_k=top_k)
                else:
                    chunks = chunks[:top_k]

            confidence = _retrieval_confidence(chunks)
        except Exception as exc:
            logger.warning("RAG retrieval failed, falling back to chat-only response: %s", exc)
            try:
                answer = await _answer_general_chat(question, session_id=session_id)
            except Exception as chat_exc:
                logger.warning("Fallback chat generation failed: %s", chat_exc)
                answer = (
                    "I hit a temporary model limit while generating this response. "
                    "Please try again in about a minute."
                )

            if session_id:
                _push_history(session_id, "user", question)
                _push_history(session_id, "assistant", answer)
            return {"answer": answer, "sources": []}

    if not chunks:
        answer = await _answer_general_chat(question, session_id=session_id)
        if session_id:
            _push_history(session_id, "user", question)
            _push_history(session_id, "assistant", answer)
        return {"answer": answer, "sources": []}

    # Low-confidence factual asks should ask a targeted clarification question,
    # not hallucinate or force a weak answer.
    if _needs_clarification(question, chunks, confidence):
        clarification = await _clarify_question(question)
        if session_id:
            _push_history(session_id, "user", question)
            _push_history(session_id, "assistant", clarification)
        return {"answer": clarification, "sources": []}

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
    messages: List[dict] = [{"role": "system", "content": _build_system_prompt()}]
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
    logger.info("Query completed via RAG path in %.2fms", (time.perf_counter() - start_ts) * 1000)

    # 8. Persist session memory
    if session_id:
        _push_history(session_id, "user", question)
        _push_history(session_id, "assistant", answer)

    return {"answer": answer, "sources": sources}
