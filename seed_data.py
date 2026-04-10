"""
Vera — Seed Data Script

Seeds your Supabase database with realistic demo leads across all pipeline stages.
Run once before recording your demo video.

Usage:
    python seed_data.py

Requires your .env file to be configured.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

load_dotenv()

from supabase import create_client

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]

sb = create_client(SUPABASE_URL, SUPABASE_KEY)

def ts(days_ago=0, hours_ago=0):
    """Return ISO timestamp offset from now."""
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    return dt.isoformat()

LEADS = [
    # ── NEW — High priority ───────────────────────────────────────────────
    {
        "id": str(uuid.uuid4()),
        "name": "Marcus Bennett",
        "email": "marcus@finovatech.io",
        "company": "Finova Tech",
        "score": 91.0,
        "priority": 3,
        "status": "new",
        "source": "website",
        "notes": "We need a full brand identity and website for our Series A launch in Q1. Team of 12, budget is $8,000–$12,000. Timeline is critical — launch date is February 15.",
        "recommended_action": "Call Marcus within 24 hours — confirmed $8–12k budget and hard Q1 deadline. Lead with your fintech case studies.",
        "created_at": ts(hours_ago=2),
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Sophie Laurent",
        "email": "sophie@paystackr.com",
        "company": "PayStackr",
        "score": 84.0,
        "priority": 3,
        "status": "new",
        "source": "referral",
        "notes": "We just closed our seed round and need to completely rebrand before we go public-facing. We've been working with an in-house designer but we need an agency now. Budget is flexible for the right team.",
        "recommended_action": "Respond within 24 hours — seed-funded, no budget ceiling mentioned, and competitor risk is high. Ask for a call this week.",
        "created_at": ts(hours_ago=5),
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Olivia Carter",
        "email": "olivia@growthlabglobal.com",
        "company": "GrowthLab Global",
        "score": 67.0,
        "priority": 2,
        "status": "new",
        "source": "website",
        "notes": "Looking for a digital agency to help us with paid social and content strategy. We're a growth consultancy based in London, team of 6.",
        "recommended_action": "Send a qualifying email to Olivia — ask about monthly budget and whether they need execution or strategy only.",
        "created_at": ts(hours_ago=8),
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Daniel Foster",
        "email": "danielfoster22@gmail.com",
        "company": "",
        "score": 22.0,
        "priority": 1,
        "status": "new",
        "source": "website",
        "notes": "Hi, how much for a website?",
        "recommended_action": "Add to low-priority nurture — no company, no budget signal, generic Gmail. Send pricing FAQ link only.",
        "created_at": ts(days_ago=1),
    },

    # ── CONTACTED ────────────────────────────────────────────────────────
    {
        "id": str(uuid.uuid4()),
        "name": "Sarah Mitchell",
        "email": "sarah@elmcommerce.com",
        "company": "Elm Commerce",
        "score": 88.0,
        "priority": 3,
        "status": "contacted",
        "source": "linkedin",
        "notes": "Ready to discuss retainer options. We have been growing fast and need ongoing content, social, and email. Budget confirmed at $3,000/month.",
        "recommended_action": "Schedule a call this week — confirmed $3k/month budget and ready to start. Bring the retainer scope document.",
        "created_at": ts(days_ago=2),
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Ethan Reynolds",
        "email": "e.reynolds@proptechnorth.com",
        "company": "PropTech North",
        "score": 61.0,
        "priority": 2,
        "status": "contacted",
        "source": "website",
        "notes": "Comparing 3 agencies for a website redesign. Will decide by end of month. Need modern design and solid SEO foundation.",
        "recommended_action": "Follow up with Ethan — send a case study showing similar website work and emphasise your SEO process.",
        "created_at": ts(days_ago=3),
    },

    # ── QUALIFIED ────────────────────────────────────────────────────────
    {
        "id": str(uuid.uuid4()),
        "name": "Lucas Meyer",
        "email": "lucas@trovehealth.com",
        "company": "Trove Health",
        "score": 93.0,
        "priority": 3,
        "status": "qualified",
        "source": "referral",
        "notes": "Proposal reviewed and they are ready to proceed with the full brand package plus 6-month retainer. Final decision on budget split pending CFO approval.",
        "recommended_action": "Send contract to Lucas today — proposal accepted, pending only internal budget approval. Strike while it's warm.",
        "created_at": ts(days_ago=5),
    },

    # ── CONVERTED ────────────────────────────────────────────────────────
    {
        "id": str(uuid.uuid4()),
        "name": "Nathan Brooks",
        "email": "nathan@boltlogistics.com",
        "company": "Bolt Logistics",
        "score": 96.0,
        "priority": 3,
        "status": "converted",
        "source": "referral",
        "notes": "Full rebrand, new website, and 12-month retainer. Contract signed. Total value $15,000.",
        "recommended_action": "Client onboarded — kickoff meeting scheduled for next Tuesday.",
        "created_at": ts(days_ago=18),
    },
    {
        "id": str(uuid.uuid4()),
        "name": "Hannah Cooper",
        "email": "h.cooper@stackpay.co",
        "company": "StackPay",
        "score": 89.0,
        "priority": 3,
        "status": "converted",
        "source": "website",
        "notes": "Brand identity project. Logo, guidelines, pitch deck template. Delivered and signed off.",
        "recommended_action": "Project complete — follow up in 60 days for potential retainer upsell.",
        "created_at": ts(days_ago=32),
    },
]


def seed():
    print("━" * 48)
    print("  Vera — Seeding demo data")
    print("━" * 48)

    # Clear existing leads (optional — comment out to keep existing data)
    existing = sb.table("leads").select("id").execute()
    if existing.data:
        print(f"  Clearing {len(existing.data)} existing leads…")
        sb.table("leads").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()

    print(f"  Inserting {len(LEADS)} leads…")
    sb.table("leads").insert(LEADS).execute()

    # Print summary
    by_status = {}
    for l in LEADS:
        by_status.setdefault(l["status"], []).append(l["name"])

    for status, names in by_status.items():
        print(f"  [{status.upper()}] {', '.join(names)}")

    print()
    print("  ✓ Done. Open your dashboard to see the pipeline.")
    print("━" * 48)


if __name__ == "__main__":
    seed()
