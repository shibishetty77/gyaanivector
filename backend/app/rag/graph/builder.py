"""
Builds the GraphStore by fetching all documents from Supabase at startup.
"""

import logging
from functools import lru_cache
from typing import Any

import networkx as nx
from supabase import Client, create_client

from app.config import get_settings
from app.rag.graph.extractor import extract_graph_elements
from app.rag.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


def build_graph_store(client: Client, table_name: str) -> GraphStore:
    """
    Fetch all documents from the vector store and build the entity-to-entity
    directed graph using the extractor rules.
    """
    response = (
        client.table(table_name)
        .select("doc_id, source_path, category, subcategory, content, metadata")
        .execute()
    )
    rows = response.data or []

    graph = nx.MultiDiGraph()
    docs_by_id: dict[str, dict[str, Any]] = {}

    for row in rows:
        # Every ingested document has a generated UUID in doc_id
        doc_id = row.get("doc_id")
        if not doc_id:
            continue
        docs_by_id[doc_id] = row

        metadata = row.get("metadata") or {}
        node_id, aliases, edges = extract_graph_elements(metadata)
        
        if not node_id:
            continue
            
        # Add or update the primary node for this document
        if not graph.has_node(node_id):
            graph.add_node(node_id, aliases=set(), doc_ids=set())
            
        graph.nodes[node_id]["aliases"].update(aliases)
        graph.nodes[node_id]["doc_ids"].add(doc_id)
        
        # Add edges and implicitly create target nodes
        for src, relation, target in edges:
            if not graph.has_node(target):
                # We might not have metadata for the target yet (or ever),
                # but we create the node so we can traverse to it.
                # The target alias defaults to its own ID.
                graph.add_node(target, aliases={target}, doc_ids=set())
            
            graph.add_edge(src, target, relation=relation)

    logger.info(
        "[GraphStore Builder] built from %d rows: %d nodes, %d edges",
        len(rows),
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    
    return GraphStore(graph=graph, docs_by_id=docs_by_id)


@lru_cache
def get_graph_store() -> GraphStore:
    """Built once per process at startup."""
    settings = get_settings()
    client: Client = create_client(settings.supabase_url, settings.supabase_service_key)
    return build_graph_store(client, settings.knowledge_table)
