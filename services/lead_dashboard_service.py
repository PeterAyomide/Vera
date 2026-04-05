"""LeadSentinel – dashboard & lead management service.

Provides CRUD, filtering, stats, bulk scoring, and AI-powered
context-based lead search.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import List

from openai import AsyncOpenAI

from services.db import supabase

logger = logging.getLogger(__name__)

_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)


_PIPELINE_STAGES = ["new", "contacted", "qualified", "converted"]


def _infer_stage_path(status: str | None) -> list[str]:
    status = (status or "new").lower()
    if status == "new":
        return ["new"]
    if status == "contacted":
        return ["new", "contacted"]
    if status == "qualified":
        return ["new", "contacted", "qualified"]
    if status == "converted":
        return ["new", "contacted", "qualified", "converted"]
    return ["new"]


def _extract_stage_history(lead: dict) -> list[str]:
    metadata = lead.get("metadata") or {}
    history = metadata.get("status_history")
    if isinstance(history, list):
        cleaned = [str(s).lower() for s in history if str(s).lower() in (_PIPELINE_STAGES + ["lost"])]
        if cleaned:
            return cleaned
    return _infer_stage_path(lead.get("status"))


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
    current = get_lead(lead_id)
    if not current:
        return {}

    if "status" in updates:
        next_status = str(updates["status"]).lower()
        prev_status = str(current.get("status") or "new").lower()

        metadata = current.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}

        history = metadata.get("status_history")
        if not isinstance(history, list):
            history = _infer_stage_path(prev_status)
        else:
            history = [str(s).lower() for s in history]

        if not history:
            history = _infer_stage_path(prev_status)

        if not history or history[-1] != next_status:
            history.append(next_status)

        metadata["status_history"] = history
        # Persist the first moment a lead becomes converted so trend charts
        # are not distorted by later unrelated updates to updated_at.
        if next_status == "converted" and prev_status != "converted":
            metadata.setdefault("converted_at", datetime.now(timezone.utc).isoformat())
        if next_status == "lost" and prev_status in _PIPELINE_STAGES:
            metadata["lost_from_stage"] = prev_status

        updates["metadata"] = metadata

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
        "id, score, priority, status, metadata, created_at", count="exact"
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

    reached_counts = {stage: 0 for stage in _PIPELINE_STAGES}
    for lead in leads:
        reached = set(_extract_stage_history(lead))
        for stage in _PIPELINE_STAGES:
            if stage in reached:
                reached_counts[stage] += 1

    contacted_rate = round((reached_counts["contacted"] / reached_counts["new"]) * 100, 1) if reached_counts["new"] else 0
    qualified_rate = round((reached_counts["qualified"] / reached_counts["contacted"]) * 100, 1) if reached_counts["contacted"] else 0
    converted_rate = round((reached_counts["converted"] / reached_counts["qualified"]) * 100, 1) if reached_counts["qualified"] else 0

    return {
        "total_leads": total,
        "average_score": avg_score,
        "highest_score": high_score,
        "conversion_rate": conversion_rate,
        "funnel": {
            "reached": reached_counts,
            "rates": {
                "contacted_from_new": contacted_rate,
                "qualified_from_contacted": qualified_rate,
                "converted_from_qualified": converted_rate,
            },
        },
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


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def get_dashboard_trends(days: int = 30, user_id: str | None = None) -> dict:
    """Return daily trend data for lead creation, conversion, and avg score."""
    days = max(7, min(days, 90))

    query = supabase.table("leads").select(
        "id, status, score, metadata, created_at, updated_at"
    )
    if user_id:
        query = query.eq("user_id", user_id)

    result = query.execute()
    leads = result.data or []

    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)

    daily = {}
    for i in range(days):
        d = start + timedelta(days=i)
        key = d.isoformat()
        daily[key] = {
            "date": key,
            "created": 0,
            "converted": 0,
            "score_sum": 0.0,
            "score_count": 0,
        }

    for lead in leads:
        created_dt = _parse_iso(lead.get("created_at"))
        if created_dt:
            created_key = created_dt.date().isoformat()
            if created_key in daily:
                daily[created_key]["created"] += 1
                score = lead.get("score")
                if score is not None:
                    daily[created_key]["score_sum"] += float(score)
                    daily[created_key]["score_count"] += 1

        if lead.get("status") == "converted":
            metadata = lead.get("metadata") or {}
            converted_dt = (
                _parse_iso(metadata.get("converted_at"))
                or _parse_iso(lead.get("updated_at"))
                or created_dt
            )
            if converted_dt:
                converted_key = converted_dt.date().isoformat()
                if converted_key in daily:
                    daily[converted_key]["converted"] += 1

    rows = []
    created_total = 0
    converted_total = 0
    score_total = 0.0
    score_count = 0
    for key in sorted(daily.keys()):
        item = daily[key]
        avg_score = (
            round(item["score_sum"] / item["score_count"], 1)
            if item["score_count"]
            else 0
        )
        rows.append(
            {
                "date": item["date"],
                "created": item["created"],
                "converted": item["converted"],
                "avg_score": avg_score,
            }
        )
        created_total += item["created"]
        converted_total += item["converted"]
        score_total += item["score_sum"]
        score_count += item["score_count"]

    return {
        "days": days,
        "summary": {
            "created": created_total,
            "converted": converted_total,
            "conversion_rate": round((converted_total / created_total) * 100, 1)
            if created_total
            else 0,
            "average_score": round(score_total / score_count, 1) if score_count else 0,
        },
        "daily": rows,
    }


def get_at_risk_leads(limit: int = 5, user_id: str | None = None) -> dict:
    """Return leads that need immediate follow-up based on priority and inactivity."""
    limit = max(1, min(limit, 20))

    query = supabase.table("leads").select(
        "id, name, company, score, priority, status, "
        "last_contacted_at, created_at"
    )
    if user_id:
        query = query.eq("user_id", user_id)

    result = query.execute()
    leads = result.data or []

    now = datetime.now(timezone.utc)
    risk_items = []

    for lead in leads:
        status = lead.get("status")
        if status in {"converted", "lost"}:
            continue

        created_dt = _parse_iso(lead.get("created_at"))
        last_contacted_dt = _parse_iso(lead.get("last_contacted_at")) or created_dt
        if not last_contacted_dt:
            continue

        age_hours = (now - last_contacted_dt).total_seconds() / 3600
        reasons = []
        risk_score = 0

        priority = int(lead.get("priority") or 0)
        lead_score = float(lead.get("score") or 0)

        if priority >= 3 and age_hours >= 24:
            reasons.append("High priority with no contact in 24h")
            risk_score += 65

        if status == "qualified" and age_hours >= 72:
            reasons.append("Qualified lead inactive for 3 days")
            risk_score += 45

        if lead_score >= 75 and age_hours >= 48:
            reasons.append("High scoring lead waiting for follow-up")
            risk_score += 30

        if not reasons:
            continue

        risk_items.append(
            {
                "id": lead.get("id"),
                "name": lead.get("name") or "Unknown",
                "company": lead.get("company") or "—",
                "status": status,
                "priority": priority,
                "score": round(lead_score),
                "hours_since_contact": int(age_hours),
                "risk_score": risk_score,
                "reasons": reasons,
            }
        )

    risk_items.sort(
        key=lambda x: (x["risk_score"], x["priority"], x["score"]), reverse=True
    )

    return {
        "count": len(risk_items),
        "leads": risk_items[:limit],
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
    from services.lead_service import analyze_lead

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
