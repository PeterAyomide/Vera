# Vera — AI Operating System for Digital Agencies

Three capabilities. One dashboard.

- **CorporateBrain** — Upload docs, ask questions in plain English, get grounded answers with source citations
- **LeadSentinel** — Every inbound lead scored 0–100 with a specific recommended action in seconds
- **Outreach Engine** — One-click personalized email drafts built from your knowledge base + the lead's context

---

## Setup (do this once, then deploy)

### Step 1 — Supabase

1. Go to [supabase.com](https://supabase.com) → New project
2. Wait for it to provision (~2 minutes)
3. Go to **Database → Extensions** → search `vector` → Enable it
4. Go to **SQL Editor → New query** → paste the entire contents of `sql/schema.sql` → Run
5. Go to **Settings → API** — copy:
   - **Project URL** → this is your `SUPABASE_URL`
   - **service_role** key (not anon) → this is your `SUPABASE_SERVICE_KEY`

### Step 2 — GitHub Token (for free GPT-4o-mini)

1. Go to github.com → Settings → Developer Settings → Personal access tokens → Tokens (classic)
2. Generate new token — no special scopes needed, just generate it
3. Copy it → this is your `GITHUB_TOKEN`

### Step 3 — Configure environment

```bash
cp .env.example .env
```

Edit `.env`:

```
SUPABASE_URL=https://your-project-ref.supabase.co
SUPABASE_SERVICE_KEY=eyJ...your service role key...
GITHUB_TOKEN=ghp_...your token...
API_KEY=make-up-any-secret-string-here
```

### Step 4 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 5 — Seed demo data (for the demo video)

```bash
python seed_data.py
```

This populates your pipeline with 9 realistic leads across all stages so the dashboard looks live.

### Step 6 — Run locally

```bash
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

Click **Configure API** in the sidebar:
- API Base URL: `http://localhost:8000`
- API Key: whatever you set as `API_KEY` in `.env`
- Agency Name: your agency's name

---

## Deploying to Railway (recommended — free tier works)

1. Push your code to a GitHub repo (make sure `.env` is in `.gitignore`)
2. Go to [railway.app](https://railway.app) → New Project → Deploy from GitHub repo
3. Select your repo
4. Go to **Variables** → add all four env vars from your `.env`
5. Railway will auto-detect Python and deploy with `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Copy your Railway URL (e.g. `https://vera-production.up.railway.app`)
7. Update `ALLOWED_ORIGINS` in Variables to your Railway URL

---

## Website embed (automatic lead capture)

Add this to any contact form page — leads flow directly into Vera:

```html
<script src="https://your-vera-url.railway.app/embed.js"></script>
```

Then edit `static/embed.js` to set your API URL, key, and form field names.

---

## Project structure

```
vera/
├── main.py                   ← FastAPI app, all routes, rate limiting
├── requirements.txt
├── seed_data.py              ← Run once to populate demo data
├── .env.example
├── services/
│   ├── ingest_service.py     ← Chunk, embed, store documents
│   ├── query_service.py      ← RAG query with session memory
│   ├── lead_service.py       ← AI lead scoring
│   ├── lead_dashboard_service.py
│   ├── chat_service.py       ← Chat session persistence
│   ├── auth.py
│   └── db.py
├── static/
│   ├── dashboard.html        ← Full dashboard (single file)
│   └── embed.js              ← Website embed snippet
└── sql/
    └── schema.sql            ← Run this in Supabase SQL Editor
```

---

## Demo video flow (record this exactly)

1. Open your deployed Vera URL
2. Go to **Lead Pipeline** — show the seeded pipeline with scores
3. Click on **Marcus Okonkwo** (score 91) — show the detail panel and AI recommended action
4. Click **Draft email** — show Outreach Engine pulling from knowledge base and generating the draft
5. Go to **Documents** — drag and drop your pricing deck or a case study
6. Go to **Knowledge Chat** → click **New Chat** → ask *"What are our retainer rates?"*
7. Show the answer with source citations
8. Go back to **Lead Pipeline** → click **Add Lead** → fill in a realistic new lead → show it get scored live
9. Show the lead card appearing in the New column

Total demo: under 3 minutes. Every feature visible.

---

## Environment variables reference

| Variable | Where to get it | Required |
|---|---|---|
| `SUPABASE_URL` | Supabase → Settings → API → Project URL | ✓ |
| `SUPABASE_SERVICE_KEY` | Supabase → Settings → API → service_role key | ✓ |
| `GITHUB_TOKEN` | GitHub → Settings → Developer Settings → Personal access tokens | ✓ |
| `API_KEY` | You make this up — any random string | ✓ |
| `ALLOWED_ORIGINS` | Your deployed frontend URL (comma-separated) | Production |

---

## Rate limits (default, change via env vars)

| Endpoint | Limit | Env var |
|---|---|---|
| `/query` | 30 req/min | `RATE_LIMIT_QUERY` |
| `/analyze-lead` | 30 req/min | `RATE_LIMIT_ANALYZE` |
| `/upload` | 10 req/min | `RATE_LIMIT_UPLOAD` |
| everything else | 120 req/min | `RATE_LIMIT_DEFAULT` |
