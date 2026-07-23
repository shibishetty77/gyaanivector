-- Run this once in the Supabase SQL editor (Project -> SQL Editor -> New query)
-- before running ingest.py.

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Table that holds one row per YAML file
create table if not exists knowledge_chunks (
    id           uuid primary key default gen_random_uuid(),
    source_path  text not null unique,      -- relative path inside the zip, e.g. backend/knowledge/facts/components/dht22.yaml
    category     text,                      -- top-level folder: facts | classes | eval
    subcategory  text,                      -- next folder down: components | platforms | sensor | bus | ...
    doc_id       text,                      -- the yaml's own id/class/evaluation_id field, if present
    content      text not null,             -- raw file text (comments included) - what got embedded
    metadata     jsonb default '{}'::jsonb, -- parsed yaml as JSON, for filtering/debugging
    embedding    vector(384),               -- all-MiniLM-L6-v2 (local, free) dimension; change if you use a different model
    updated_at   timestamptz not null default now()
);

-- 3. Index for fast approximate nearest-neighbour search
create index if not exists knowledge_chunks_embedding_idx
    on knowledge_chunks
    using ivfflat (embedding vector_cosine_ops)
    with (lists = 100);

-- 4. Helper function for similarity search (used by RAG queries later)
create or replace function match_knowledge_chunks (
    query_embedding vector(384),
    match_count int default 5,
    filter_category text default null
)
returns table (
    id uuid,
    source_path text,
    category text,
    subcategory text,
    doc_id text,
    content text,
    metadata jsonb,
    similarity float
)
language sql stable
as $$
    select
        id, source_path, category, subcategory, doc_id, content, metadata,
        1 - (embedding <=> query_embedding) as similarity
    from knowledge_chunks
    where filter_category is null or category = filter_category
    order by embedding <=> query_embedding
    limit match_count;
$$;