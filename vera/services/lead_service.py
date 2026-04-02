"""LeadSentinel – AI-powered lead scoring service.

Sends lead data to an OpenAI LLM and returns a structured score.
"""

from __future__ import annotations

import json
import logging
import os

from openai import AsyncOpenAI

from core_engine.models import LeadScore
from core_engine.services.db import supabase

logger = logging.getLogger(__name__)

# ── GitHub Models client (OpenAI-compatible) ────────────────────────────────
_openai = AsyncOpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ.get("GITHUB_TOKEN", ""),
)

_SYSTEM_PROMPT = """\
You are **Aria**, the AI lead-scoring analyst at a digital agency. \
You score inbound leads with the precision of a senior sales manager who has reviewed \
thousands of B2B enquiries.

Given a lead's details, return a JSON object with exactly three keys:
  "score"              – number 0-100 (100 = highest buying intent, 0 = clearly irrelevant)
  "priority"           – integer: 1 (low), 2 (medium), 3 (high)
  "recommended_action" – a single, specific, actionable sentence for the sales team to act on NOW

SCORING FRAMEWORK:
  Email domain quality (0-20 pts):
    • Branded corporate domain (e.g. name@brandco.com)   → 15-20
    • Ambiguous / non-obvious domain                       → 8-14
    • Free provider (gmail, hotmail, yahoo, etc.)          → 0-7

  Message intent & clarity (0-40 pts):
    • Specific problem stated + budget/timeline mentioned  → 35-40
    • Clear interest with some detail                      → 20-34
    • Vague enquiry or generic question                    → 5-19
    • Spam, test, or completely off-topic                  → 0-4

  Company signals (0-25 pts):
    • Named company, identifiable size, clear vertical     → 20-25
    • Company name present but limited detail              → 10-19
    • No company or "self-employed" solo contact           → 0-9

  Urgency / buying signals (0-15 pts):
    • Explicit deadline, RFP, or "ready to start"          → 12-15
    • Comparison shopping / evaluating options             → 6-11
    • No urgency signal                                    → 0-5

RECOMMENDED ACTION:
  • score ≥ 70  → "Call [Name] within 24 hours — they have [specific buying signal]."
  • score 40-69 → "Send a qualifying email to [Name] at [Company] — ask [specific question]."
  • score < 40  → "Add to low-priority nurture sequence — no immediate action required."
  Always be specific about the company or message content in the action.

Return ONLY valid JSON — no markdown fences, no extra text.
"""


async def analyze_lead(
    lead_id: str,
    name: str,
    email: str,
    company: str,
    message: str,
    user_id: str | None = None,
) -> LeadScore:
    """Score a lead via OpenAI and persist the result to Supabase."""

    user_content = (
        f"Name: {name}\n"
        f"Email: {email}\n"
        f"Company: {company}\n"
        f"Message: {message}"
    )

    logger.info("Scoring lead %s (%s @ %s)", lead_id, name, company)

    completion = await _openai.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )

    raw = completion.choices[0].message.content
    data = json.loads(raw)  # type: ignore[arg-type]
    scoring = LeadScore(**data)

    # Persist to Supabase
    supabase.table("leads").update(
        {
            "score": scoring.score,
            "priority": scoring.priority,
            "recommended_action": scoring.recommended_action,
        }
    ).eq("id", lead_id).execute()

    logger.info(
        "Lead %s scored: %d (%s) – %s",
        lead_id,
        scoring.score,
        scoring.priority,
        scoring.recommended_action,
    )
    return scoring
