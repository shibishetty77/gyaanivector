"""
LangChain retriever backed by the existing Supabase pgvector setup, with an
optional knowledge-graph expansion pass layered on top.

Flow: question -> local embedding -> match_knowledge_chunks RPC -> Documents
                                                                  -> (optional)
                                                                     1-hop graph
                                                                     expansion
No documents are ever (re-)embedded here -- only the incoming query.

The graph expansion step is purely additive: it never removes a vector hit,
it only adds documents that are directly connected (via `class:`,
`facts_used:`, or `classes_used:`) to something vector search already
surfaced -- e.g. guaranteeing a component's composed class rule-pack is
present even if its own embedding similarity ranked it low. See
knowledge_graph.py for how the graph itself is built.
"""

import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
from pydantic import ConfigDict

from app.rag.embeddings import LocalQueryEmbeddings
from app.rag.knowledge_graph import KnowledgeGraph
from app.rag.vector_store import SupabaseVectorStore

logger = logging.getLogger(__name__)


class SupabaseRAGRetriever(BaseRetriever):
    """
    A LangChain retriever that:
      1. Embeds the user's question locally (SentenceTransformer).
      2. Calls the existing `match_knowledge_chunks` Supabase RPC function.
      3. Wraps each returned row as a LangChain `Document`.
      4. (Optional) Expands the result set by 1 hop through the in-memory
         knowledge graph, adding any directly-composed/related documents
         that vector search alone didn't surface.

    Optional `category` filter maps straight to the RPC's `filter_category`
    argument (e.g. "facts", "classes", "eval").
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: LocalQueryEmbeddings
    vector_store: SupabaseVectorStore
    k: int = 5
    category: str | None = None
    knowledge_graph: KnowledgeGraph | None = None
    graph_expansion_hops: int = 1

    def _rows_to_documents(self, rows: list[dict]) -> list[Document]:
        documents = []
        for row in rows:
            metadata = {
                "source_path": row.get("source_path"),
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "doc_id": row.get("doc_id"),
                "similarity": row.get("similarity"),
                "retrieved_via": "vector",
            }
            documents.append(Document(page_content=row.get("content", ""), metadata=metadata))
        return documents

    def _expand_with_graph(self, documents: list[Document]) -> list[Document]:
        if self.knowledge_graph is None:
            return documents

        seed_ids = [d.metadata.get("doc_id") for d in documents if d.metadata.get("doc_id")]
        expanded_ids = self.knowledge_graph.expand(seed_ids, hops=self.graph_expansion_hops)

        existing_ids = set(seed_ids)
        added: list[Document] = []
        for doc_id in expanded_ids:
            if doc_id in existing_ids:
                continue
            row = self.knowledge_graph.get_document(doc_id)
            if row is None:
                continue
            metadata = {
                "source_path": row.get("source_path"),
                "category": row.get("category"),
                "subcategory": row.get("subcategory"),
                "doc_id": row.get("doc_id"),
                "similarity": None,
                "retrieved_via": "graph",
            }
            added.append(Document(page_content=row.get("content", ""), metadata=metadata))

        if added:
            logger.info(
                "[STAGE: Retriever] graph expansion added %d document(s) | ids=%s",
                len(added), [d.metadata.get("doc_id") for d in added],
            )

        return documents + added

    def _log_retriever_stages(self, documents: list[Document]) -> None:
        doc_ids = [d.metadata.get("doc_id") or d.metadata.get("source_path") for d in documents]
        logger.info("[STAGE: Retriever] documents before filtering | count=%d ids=%s", len(documents), doc_ids)
        logger.info("[STAGE: Retriever] documents after filtering | count=%d (no filtering logic present)", len(documents))
        logger.info("[STAGE: Retriever] documents after threshold | count=%d (no threshold logic present)", len(documents))
        logger.info("[STAGE: Retriever] documents after deduplication | count=%d (no dedup logic present)", len(documents))
        logger.info("[STAGE: Retriever] documents sent to chain | count=%d ids=%s", len(documents), doc_ids)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        logger.info("[STAGE: Retriever] entered stage | query_preview=%r category=%r k=%d", query[:80], self.category, self.k)
        query_embedding = self.embeddings.embed_query(query)
        rows = self.vector_store.similarity_search(query_embedding, k=self.k, category=self.category)
        documents = self._rows_to_documents(rows)
        documents = self._expand_with_graph(documents)
        self._log_retriever_stages(documents)
        logger.info("[STAGE: Retriever] completed stage | documents_returned=%d", len(documents))
        return documents

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        logger.info("[STAGE: Retriever] entered stage (async) | query_preview=%r category=%r k=%d", query[:80], self.category, self.k)
        query_embedding = await self.embeddings.aembed_query(query)
        rows = await self.vector_store.asimilarity_search(query_embedding, k=self.k, category=self.category)
        documents = self._rows_to_documents(rows)
        documents = self._expand_with_graph(documents)
        self._log_retriever_stages(documents)
        logger.info("[STAGE: Retriever] completed stage (async) | documents_returned=%d", len(documents))
        return documents


def build_retriever(
    embeddings: LocalQueryEmbeddings,
    vector_store: SupabaseVectorStore,
    k: int,
    category: str | None = None,
    knowledge_graph: KnowledgeGraph | None = None,
) -> SupabaseRAGRetriever:
    """Factory -- a fresh retriever per request when a category filter is requested,
    since `category` is per-query, not a fixed app-wide setting."""
    return SupabaseRAGRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        k=k,
        category=category,
        knowledge_graph=knowledge_graph,
    )
