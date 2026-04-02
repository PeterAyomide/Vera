"""LeadSentinel – dashboard & lead management service.

Provides CRUD, filtering, stats, bulk scoring, and AI-powered
context-based lead search.
"""

from __future__ import annotations

import json
import logging
import os
from typing import List

from openai import AsyncOpenAI

from core_engine.services.db import supabase

logger = logging.getLogger(__name__)

_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)


# ── List / Filter Leads ─────────────────────────────────────────────────────

def list_leads(
    user_id: str | None = None,
    status: str | None = None,
    priority_min: int | None = None,
    score_min: float | None = None,
    search: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    page: int = 1,
    page_size: int = 25,
) -> dict:
    """List leads with filtering, sorting, and pagination."""

    query = supabase.table("leads").select(
        "id, name, email, company, phone, score, priority, status, "
        "source, notes, recommended_action, metadata, "
        "last_contacted_at, created_at, updated_at",
        count="exact",
    )

    if user_id:
        query = query.eq("user_id", user_id)
    if status:
        query = query.eq("status", status)
    if priority_min is not None:
        query = query.gte("priority", priority_min)
    if score_min is not None:
        query = query.gte("score", score_min)
    if search:
        # Full-text search across name, email, company
        query = query.or_(
            f"name.ilike.%{search}%,"
            f"email.ilike.%{search}%,"
            f"company.ilike.%{search}%"
        )

    # Sorting
    ascending = sort_order.lower() == "asc"
    query = query.order(sort_by, desc=not ascending)

    # Pagination
    offset = (page - 1) * page_size
    query = query.range(offset, offset + page_size - 1)

    result = query.execute()

    return {
        "leads": result.data or [],
        "total": result.count or 0,
        "page": page,
        "page_size": page_size,
    }


# ── Get Single Lead ─────────────────────────────────────────────────────────

def get_lead(lead_id: str) -> dict | None:
    """Get a single lead by ID."""
    result = (
        supabase.table("leads")
        .select("*")
        .eq("id", lead_id)
        .single()
        .execute()
    )
    return result.data


# ── Update Lead ──────────────────────────────────────────────────────────────

def update_lead(lead_id: str, updates: dict) -> dict:
    """Update lead fields (status, notes, etc.)."""
    result = (
        supabase.table("leads")
        .update(updates)
        .eq("id", lead_id)
        .execute()
    )
    return result.data[0] if result.data else {}


# ── Dashboard Stats ──────────────────────────────────────────────────────────

def get_dashboard_stats(user_id: str | None = None) -> dict:
    """Get aggregate stats for the dashboard."""

    query = supabase.table("leads").select(
        "id, score, priority, status, created_at", count="exact"
    )
    if user_id:
        query = query.eq("user_id", user_id)

    result = query.execute()
    leads = result.data or []
    total = result.count or 0

    # Compute stats
    scores = [l["score"] for l in leads if l.get("score") is not None]
    avg_score = round(sum(scores) / len(scores), 1) if scores else 0
    high_score = max(scores) if scores else 0

    status_counts = {}
    for l in leads:
        s = l.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    priority_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for l in leads:
        p = l.get("priority", 0) or 0
        priority_counts[p] = priority_counts.get(p, 0) + 1

    # Conversion rate
    converted = status_counts.get("converted", 0)
    conversion_rate = round((converted / total) * 100, 1) if total > 0 else 0

    return {
        "total_leads": total,
        "average_score": avg_score,
        "highest_score": high_score,
        "conversion_rate": conversion_rate,
        "by_status": status_counts,
        "by_priority": {
            "none": priority_counts[0],
            "low": priority_counts[1],
            "medium": priority_counts[2],
            "high": priority_counts[3],
        },
        "hot_leads": len([s for s in scores if s >= 70]),
        "warm_leads": len([s for s in scores if 40 <= s < 70]),
        "cold_leads": len([s for s in scores if s < 40]),
    }


# ── Context-based Lead Search ───────────────────────────────────────────────

_SEARCH_PROMPT = """\
You are LeadSentinel's search AI.  Given a natural-language query and a list
of leads (JSON), return a JSON object with:
  • "matching_lead_ids" – array of lead UUIDs that best match the query
  • "explanation" – short explanation of why these leads match

Evaluate leads on ALL available fields: name, email domain, company, notes,
score, priority, status, source, and metadata.  Be generous but relevant.

Return ONLY valid JSON – no markdown fences.
"""


async def search_leads_by_context(
    query_text: str,
    user_id: str | None = None,
    limit: int = 50,
) -> dict:
    """Use AI to find leads matching a natural-language description.

    e.g. "fintech companies with high buying intent"
    """

    # Fetch candidate leads
    q = supabase.table("leads").select(
        "id, name, email, company, score, priority, status, "
        "source, notes, recommended_action, metadata"
    )
    if user_id:
        q = q.eq("user_id", user_id)
    q = q.limit(limit)
    result = q.execute()
    candidates = result.data or []

    if not candidates:
        return {"matches": [], "explanation": "No leads found in the database.", "total_searched": 0}

    # Ask the LLM to filter
    completion = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SEARCH_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Query: {query_text}\n\n"
                    f"Leads:\n{json.dumps(candidates, default=str)}"
                ),
            },
        ],
    )

    raw = completion.choices[0].message.content
    data = json.loads(raw)
    matching_ids = set(data.get("matching_lead_ids", []))

    matches = [l for l in candidates if l["id"] in matching_ids]

    return {
        "matches": matches,
        "explanation": data.get("explanation", ""),
        "total_searched": len(candidates),
    }


# ── Bulk Score ───────────────────────────────────────────────────────────────

async def bulk_score_leads(
    user_id: str | None = None,
    limit: int = 20,
) -> dict:
    """Score all unscored (score=0 or NULL) leads using asyncio.gather."""
    import asyncio
    from core_engine.services.lead_service import analyze_lead

    q = supabase.table("leads").select("id, name, email, company, notes, message, user_id")
    if user_id:
        q = q.eq("user_id", user_id)
    q = q.or_("score.is.null,score.eq.0").limit(limit)

    result = q.execute()
    leads = result.data or []

    async def _score_one(lead: dict) -> bool:
        """Score a single lead; returns True on success."""
        try:
            await analyze_lead(
                lead_id=lead["id"],
                name=lead.get("name", ""),
                email=lead.get("email", ""),
                company=lead.get("company", ""),
                message=lead.get("notes") or lead.get("message", ""),
                user_id=lead.get("user_id"),
            )
            return True
        except Exception:
            logger.exception("Bulk score failed for lead %s", lead["id"])
            return False

    results = await asyncio.gather(*[_score_one(l) for l in leads])

    scored = sum(1 for r in results if r)
    errors = sum(1 for r in results if not r)

    return {
        "total_found": len(leads),
        "scored": scored,
        "errors": errors,
    }
