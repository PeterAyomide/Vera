-- ============================================================
-- Vera — Complete Database Schema
-- Run this entire file in Supabase SQL Editor
-- Requires: pgvector extension enabled (Extensions → vector)
-- ============================================================

-- Enable pgvector (safe to run even if already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- ENUMS
-- ============================================================

DO $$ BEGIN
  CREATE TYPE document_status AS ENUM ('pending', 'processing', 'completed', 'failed');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

DO $$ BEGIN
  CREATE TYPE lead_status AS ENUM ('new', 'contacted', 'qualified', 'converted', 'lost');
EXCEPTION WHEN duplicate_object THEN NULL; END $$;

-- ============================================================
-- DOCUMENTS TABLE (CorporateBrain — knowledge base)
-- ============================================================

CREATE TABLE IF NOT EXISTS documents (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL,
    status      document_status DEFAULT 'pending',
    mime_type   TEXT,
    file_size   INTEGER,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at DESC);

-- ============================================================
-- CHUNKS TABLE (embeddings for vector search)
-- ============================================================

CREATE TABLE IF NOT EXISTS chunks (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content     TEXT NOT NULL,
    embedding   vector(1536) NOT NULL,
    chunk_index INTEGER,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON chunks(document_id);

-- HNSW index for fast approximate nearest-neighbor vector search
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- ============================================================
-- LEADS TABLE (LeadSentinel — pipeline)
-- ============================================================

CREATE TABLE IF NOT EXISTS leads (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    email               TEXT,
    company             TEXT,
    phone               TEXT,
    score               NUMERIC(5,2) DEFAULT 0.0,
    priority            INTEGER DEFAULT 1,
    status              lead_status DEFAULT 'new',
    source              TEXT DEFAULT 'manual',
    notes               TEXT,
    recommended_action  TEXT,
    metadata            JSONB DEFAULT '{}',
    last_contacted_at   TIMESTAMPTZ,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_leads_status    ON leads(status);
CREATE INDEX IF NOT EXISTS idx_leads_score     ON leads(score DESC);
CREATE INDEX IF NOT EXISTS idx_leads_priority  ON leads(priority DESC);
CREATE INDEX IF NOT EXISTS idx_leads_created_at ON leads(created_at DESC);

-- ============================================================
-- CHAT SESSIONS TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title       TEXT NOT NULL DEFAULT 'New conversation',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions(updated_at DESC);

-- ============================================================
-- CHAT MESSAGES TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    sources     JSONB DEFAULT '[]',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id, created_at);

-- ============================================================
-- AUTO updated_at TRIGGER
-- ============================================================

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trg_leads_updated_at ON leads;
CREATE TRIGGER trg_leads_updated_at
    BEFORE UPDATE ON leads
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

DROP TRIGGER IF EXISTS trg_chat_sessions_updated_at ON chat_sessions;
CREATE TRIGGER trg_chat_sessions_updated_at
    BEFORE UPDATE ON chat_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

-- ============================================================
-- VECTOR SIMILARITY SEARCH FUNCTION
-- ============================================================

CREATE OR REPLACE FUNCTION match_knowledge(
    query_embedding   vector(1536),
    match_threshold   float    DEFAULT 0.3,
    match_count       int      DEFAULT 7,
    filter_user_id    uuid     DEFAULT NULL  -- kept for compatibility, unused in single-tenant
)
RETURNS TABLE (
    id          uuid,
    document_id uuid,
    content     text,
    metadata    jsonb,
    similarity  float
)
LANGUAGE plpgsql AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.document_id,
        c.content,
        c.metadata,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM chunks c
    WHERE 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- ============================================================
-- DONE
-- ============================================================

DO $$
BEGIN
  RAISE NOTICE '✓ Vera schema ready.';
  RAISE NOTICE '  Tables: documents, chunks, leads, chat_sessions, chat_messages';
  RAISE NOTICE '  Functions: match_knowledge()';
  RAISE NOTICE '  Indexes: HNSW vector index on chunks.embedding';
END $$;
