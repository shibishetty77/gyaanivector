"""
Wrapper around a NetworkX MultiDiGraph that supports querying and traversal.
"""

import logging
import re
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)


class GraphStore:
    """
    In-memory directed knowledge graph.
    Nodes represent entities (e.g., 'dht11', 'sensor').
    Edges represent typed relationships (e.g., 'is_a', 'tests_fact').
    """

    def __init__(self, graph: nx.MultiDiGraph, docs_by_id: dict[str, dict[str, Any]]) -> None:
        self._graph = graph
        self._docs_by_id = docs_by_id
        
        # Build an index mapping alias string -> list of entity IDs
        self._alias_index: dict[str, set[str]] = {}
        for node_id, data in self._graph.nodes(data=True):
            aliases = data.get("aliases", [])
            for alias in aliases:
                self._alias_index.setdefault(alias, set()).add(node_id)
                
        logger.info(
            "[GraphStore] initialized with %d nodes, %d edges, %d aliases",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(self._alias_index),
        )

    def extract_entities_from_query(self, query: str) -> set[str]:
        """
        Fast keyword matching to find known graph entities mentioned in the query.
        Uses word boundaries to prevent substring matches (e.g., 'i2c' matching 'hi2c').
        """
        found_entities = set()
        query_lower = query.lower()
        
        for alias, node_ids in self._alias_index.items():
            # Escape regex characters in the alias
            escaped_alias = re.escape(alias)
            # Match only on word boundaries
            pattern = rf"\b{escaped_alias}\b"
            if re.search(pattern, query_lower):
                found_entities.update(node_ids)
                
        return found_entities

    def traverse(self, seed_entities: set[str], max_depth: int = 1) -> dict[str, float]:
        """
        Traverse the graph starting from seed_entities up to max_depth hops.
        Returns a dictionary mapping visited entity_id -> score (higher is closer).
        """
        if not seed_entities:
            return {}

        visited: dict[str, float] = {}
        # frontier is a list of (node_id, current_depth)
        frontier = [(node, 0) for node in seed_entities if self._graph.has_node(node)]

        while frontier:
            current_node, depth = frontier.pop(0)
            
            # Score decays based on depth (e.g., depth 0 = 1.0, depth 1 = 0.5)
            score = 1.0 / (depth + 1)
            
            if current_node not in visited or visited[current_node] < score:
                visited[current_node] = score
            
            if depth < max_depth:
                # Traverse outgoing edges
                for _, neighbor, _ in self._graph.out_edges(current_node, data=True):
                    frontier.append((neighbor, depth + 1))
                # Traverse incoming edges (undirected traversal for related context)
                for neighbor, _, _ in self._graph.in_edges(current_node, data=True):
                    frontier.append((neighbor, depth + 1))

        return visited

    def get_documents_for_entities(self, entity_scores: dict[str, float]) -> dict[str, float]:
        """
        Given a map of entity_id -> score, find all documents that define or mention
        those entities and return a map of doc_id -> max_score.
        """
        doc_scores: dict[str, float] = {}
        
        for entity_id, score in entity_scores.items():
            if not self._graph.has_node(entity_id):
                continue
                
            node_data = self._graph.nodes[entity_id]
            doc_ids = node_data.get("doc_ids", set())
            
            for doc_id in doc_ids:
                if doc_id not in doc_scores or doc_scores[doc_id] < score:
                    doc_scores[doc_id] = score
                    
        return doc_scores

    def get_document(self, doc_id: str) -> dict[str, Any] | None:
        """Fetch the raw document dict (from Supabase ingestion)."""
        return self._docs_by_id.get(doc_id)
