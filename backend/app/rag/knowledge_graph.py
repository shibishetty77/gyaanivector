"""
Lightweight, in-memory knowledge graph built from relationships that already
exist in your YAML data — the `class:` composition tags on components, and
the `facts_used:` / `classes_used:` fields on eval files. No graph database,
no new Supabase schema, no re-ingestion: this is built by reading the
`metadata` jsonb column your existing ingestion pipeline already populates.

Why this exists: plain vector similarity scores structured, terse YAML
inconsistently (see `lcd-1602-i2c.yaml` scoring below unrelated components
for an LCD-wiring question, despite being composed with the exact `display`
and `i2c` class rule-packs that answer the question). This module encodes
the *authored* relationships your data already has explicitly, so retrieval
can guarantee those connected documents are present regardless of where
vector similarity happened to rank them.

This is intentionally simple: one undirected graph, one BFS-style expansion
by hop count. It is not a replacement for vector search — see
`retriever.py`, which uses vector search as the primary signal and this
graph purely to expand/augment the result set.
"""

import logging
from functools import lru_cache
from typing import Any

import networkx as nx
from supabase import create_client, Client

from app.config import get_settings

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """Undirected graph of doc_id <-> doc_id relationships, plus a cache of
    each doc_id's full row (so expansion never needs an extra Supabase call)."""

    def __init__(self, graph: nx.Graph, docs_by_id: dict[str, dict[str, Any]]) -> None:
        self._graph = graph
        self._docs_by_id = docs_by_id

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        return self._docs_by_id.get(doc_id)

    def expand(self, seed_doc_ids: list[str], hops: int = 1) -> list[str]:
        """Return doc_ids reachable from the seeds within `hops` edges,
        including the seeds themselves. Order is not significant."""
        frontier: set[str] = {d for d in seed_doc_ids if d in self._graph}
        visited: set[str] = set(frontier)

        for _ in range(hops):
            next_frontier: set[str] = set()
            for node in frontier:
                next_frontier.update(self._graph.neighbors(node))
            next_frontier -= visited
            visited |= next_frontier
            frontier = next_frontier
            if not frontier:
                break

        return list(visited)

    @property
    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def edge_count(self) -> int:
        return self._graph.number_of_edges()


def _as_list(value: Any) -> list[str]:
    """class:/facts_used:/classes_used: can be a single string or a list in
    the YAML — normalize to a list either way."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def build_knowledge_graph(client: Client, table_name: str) -> KnowledgeGraph:
    """
    One-time read of every row's doc_id/category/metadata (already stored by
    the existing ingestion pipeline — nothing new is written to Supabase).
    Builds edges from:
      - a fact's `class:` field       -> its composed class rule-pack(s)
      - an eval's `facts_used:` field  -> the facts it exercises
      - an eval's `classes_used:` field -> the classes it exercises
    """
    response = (
        client.table(table_name)
        .select("doc_id, source_path, category, subcategory, content, metadata")
        .execute()
    )
    rows = response.data or []

    graph = nx.Graph()
    docs_by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        docs_by_id[doc_id] = row
        graph.add_node(doc_id, category=row.get("category"))

    dangling_edges: list[tuple[str, str]] = []

    for row in rows:
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        metadata = row.get("metadata") or {}

        related_ids: list[str] = []
        related_ids += _as_list(metadata.get("class"))
        related_ids += _as_list(metadata.get("facts_used"))
        related_ids += _as_list(metadata.get("classes_used"))

        for related_id in related_ids:
            if related_id == doc_id:
                continue
            if related_id in docs_by_id:
                graph.add_edge(doc_id, related_id)
            else:
                # A referenced id doesn't match any ingested document — data
                # inconsistency in the YAML itself (e.g. a typo'd class name),
                # not a bug in this code. Logged, not raised, so one bad
                # reference doesn't take down graph construction.
                dangling_edges.append((doc_id, related_id))

    if dangling_edges:
        logger.warning(
            "[Knowledge graph] %d dangling reference(s) found (referenced id has no "
            "matching document) -- these are data issues in the source YAML, not "
            "code bugs: %s",
            len(dangling_edges), dangling_edges,
        )

    logger.info(
        "[Knowledge graph] built | nodes=%d edges=%d dangling_refs=%d",
        graph.number_of_nodes(), graph.number_of_edges(), len(dangling_edges),
    )

    return KnowledgeGraph(graph=graph, docs_by_id=docs_by_id)


@lru_cache
def get_knowledge_graph() -> KnowledgeGraph:
    """Built once per process at startup, from the same Supabase project as
    the vector store — a separate lightweight client is created here rather
    than reusing SupabaseVectorStore's, since this module only needs plain
    table reads, not the RPC-calling interface."""
    settings = get_settings()
    client: Client = create_client(settings.supabase_url, settings.supabase_service_key)
    return build_knowledge_graph(client, settings.knowledge_table)
