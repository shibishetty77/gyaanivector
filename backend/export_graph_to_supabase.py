"""
One-off / re-runnable script to export the in-memory knowledge graph
(built by GraphStore from the YAML files) into two Supabase tables:

    entities   -- one row per graph node
    relations  -- one row per graph edge

Run this from the `backend/` directory so relative imports resolve the
same way they do when the FastAPI app starts up:

    cd backend
    python export_graph_to_supabase.py

Prerequisites:
    1. `entities` and `relations` tables already created in Supabase (see below).
    2. `pip install supabase --break-system-packages` if not already installed.
    3. .env already has SUPABASE_URL / SUPABASE_SERVICE_KEY (same ones app.config uses).

SQL to create the tables (run once in Supabase SQL Editor) if you haven't already:

    create table if not exists entities (
        id text primary key,
        category text,
        label text,
        aliases text[],
        source_path text
    );

    create table if not exists relations (
        id serial primary key,
        source_id text references entities(id),
        target_id text references entities(id),
        relation text,
        label text
    );

This script is safe to re-run: it clears both tables and re-inserts fresh
rows every time, so it always reflects the current state of your YAML-derived
graph. Re-run it any time your ingestion / YAML files change.
"""

import logging

from supabase import Client, create_client

from app.config import get_settings
from app.rag.graph import get_graph_store  # same factory main.py uses at startup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Category resolution — mirrors node_category() in app/api/graph_routes.py
# so the exported data matches what your D3 viewer already shows.
# ---------------------------------------------------------------------------
def node_category(graph, docs_by_id, node_id: str) -> str:
    data = graph.nodes[node_id]
    doc_ids = data.get("doc_ids", set())
    for doc_id in doc_ids:
        row = docs_by_id.get(doc_id)
        if row:
            return row.get("category") or "entity"
    return "entity"


def node_sources(docs_by_id, doc_ids) -> list[str]:
    sources = []
    for did in doc_ids:
        row = docs_by_id.get(did)
        if row:
            sp = row.get("source_path", "")
            if sp:
                sources.append(sp)
    return sources


def main() -> None:
    settings = get_settings()
    supabase: Client = create_client(settings.supabase_url, settings.supabase_service_key)

    logger.info("Building GraphStore from YAML files (same as app startup)...")
    gs = get_graph_store()
    g = gs._graph
    docs_by_id = gs._docs_by_id

    logger.info("Graph has %d nodes, %d edges.", g.number_of_nodes(), g.number_of_edges())

    # -----------------------------------------------------------------
    # Build entity rows
    # -----------------------------------------------------------------
    entity_rows = []
    for node_id, data in g.nodes(data=True):
        doc_ids = list(data.get("doc_ids", set()))
        aliases = list(data.get("aliases", set()))
        label = node_id[:20] if len(node_id) <= 20 else node_id[:17] + "…"

        entity_rows.append({
            "id": node_id,
            "category": node_category(g, docs_by_id, node_id),
            "label": label,
            "aliases": aliases,
            "source_path": ",".join(node_sources(docs_by_id, doc_ids)) or None,
        })

    # -----------------------------------------------------------------
    # Build relation rows
    # -----------------------------------------------------------------
    relation_rows = []
    for src, dst, data in g.edges(data=True):
        relation = data.get("relation", "related")
        relation_rows.append({
            "source_id": src,
            "target_id": dst,
            "relation": relation,
            "label": relation.replace("_", " "),
        })

    # -----------------------------------------------------------------
    # Clear + insert. Relations first (FK-safe order on delete: relations
    # then entities; on insert: entities then relations).
    # -----------------------------------------------------------------
    logger.info("Clearing existing rows...")
    supabase.table("relations").delete().neq("id", -1).execute()
    supabase.table("entities").delete().neq("id", "").execute()

    logger.info("Inserting %d entities...", len(entity_rows))
    # Supabase/PostgREST insert has a payload size limit — batch it.
    BATCH = 500
    for i in range(0, len(entity_rows), BATCH):
        supabase.table("entities").insert(entity_rows[i:i + BATCH]).execute()

    logger.info("Inserting %d relations...", len(relation_rows))
    for i in range(0, len(relation_rows), BATCH):
        supabase.table("relations").insert(relation_rows[i:i + BATCH]).execute()

    logger.info("Done. entities=%d relations=%d", len(entity_rows), len(relation_rows))


if __name__ == "__main__":
    main()