"""
LangChain BaseRetriever implementation for Graph RAG.
"""

import asyncio
import logging
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from app.rag.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)


class GraphRetriever(BaseRetriever):
    """
    Retrieves documents by identifying entities in the query and traversing the graph.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = 5
    max_depth: int = 2
    category: str | None = None

    _graph_store: GraphStore

    def __init__(
        self,
        graph_store: GraphStore,
        k: int = 5,
        max_depth: int = 2,
        category: str | None = None,
    ) -> None:
        super().__init__(k=k, max_depth=max_depth, category=category)
        self._graph_store = graph_store

    def _query(self, query: str) -> list[Document]:
        # 1. Extract entities
        entities = self._graph_store.extract_entities_from_query(query)
        if not entities:
            logger.info("[GraphRetriever] No entities found in query: %r", query[:60])
            return []

        # 2. Traverse graph
        entity_scores = self._graph_store.traverse(entities, max_depth=self.max_depth)
        
        # 3. Retrieve document scores
        doc_scores = self._graph_store.get_documents_for_entities(entity_scores)
        if not doc_scores:
            return []

        # 4. Construct and sort Documents
        scored_docs = []
        for doc_id, score in doc_scores.items():
            row = self._graph_store.get_document(doc_id)
            if not row:
                continue
                
            # Filter by category if requested
            if self.category and row.get("category") != self.category:
                continue
                
            doc = Document(
                page_content=row.get("content", ""),
                metadata={
                    "source_path": row.get("source_path"),
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "doc_id": doc_id,
                    "similarity": None,
                    "graph_score": score,
                    "retrieved_via": "graph",
                },
            )
            scored_docs.append((score, doc))

        # Sort by graph score descending, truncate to k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for _, doc in scored_docs[:self.k]]

        logger.info(
            "[GraphRetriever] query=%r | entities=%s | returned=%d",
            query[:60], entities, len(top_docs)
        )
        return top_docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._query(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Offload traversal to a thread
        return await asyncio.to_thread(self._query, query)
