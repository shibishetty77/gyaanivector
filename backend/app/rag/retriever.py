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

from app.rag.bm25_retriever import BM25Retriever
from app.rag.embeddings import LocalQueryEmbeddings
from app.rag.graph import GraphRetriever, GraphStore
from app.rag.hybrid_retriever import HybridRetriever
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
    retrieval_mode: str = "vector",
    rrf_k: int = 60,
    graph_store: GraphStore | None = None,
) -> SupabaseRAGRetriever | BM25Retriever | GraphRetriever | HybridRetriever:
    """Factory that returns the correct retriever based on RETRIEVAL_MODE.

    Modes:
      - ``"vector"``       — dense pgvector (default, fully backward-compatible).
      - ``"bm25"``         — sparse BM25 only.
      - ``"hybrid"``       — RRF fusion of vector + BM25.
      - ``"graph"``        — entity extraction + graph traversal.
      - ``"hybrid_graph"`` — RRF fusion of vector + BM25 + graph.

    BM25/hybrid modes require ``knowledge_graph`` (carries the corpus).
    Graph modes require ``graph_store``.
    Falls back to ``"vector"`` with a warning if a required dependency is None.
    """
    vector_retriever = SupabaseRAGRetriever(
        embeddings=embeddings,
        vector_store=vector_store,
        k=k,
        category=category,
        knowledge_graph=knowledge_graph,
    )

    if retrieval_mode == "vector":
        return vector_retriever

    # --- BM25 corpus: needs knowledge_graph -----------------------------------
    bm25_retriever: BM25Retriever | None = None
    if knowledge_graph is not None:
        bm25_retriever = BM25Retriever(
            docs_by_id=knowledge_graph._docs_by_id,
            k=k,
            category=category,
        )
    else:
        if retrieval_mode in ("bm25", "hybrid", "hybrid_graph"):
            logger.warning(
                "[build_retriever] retrieval_mode=%r requires knowledge_graph but it is None. "
                "Falling back to 'vector'.",
                retrieval_mode,
            )
            return vector_retriever

    if retrieval_mode == "bm25":
        logger.info("[build_retriever] mode=bm25 (category=%r k=%d)", category, k)
        return bm25_retriever  # type: ignore[return-value]

    if retrieval_mode == "hybrid":
        logger.info("[build_retriever] mode=hybrid (category=%r k=%d rrf_k=%d)", category, k, rrf_k)
        return HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,  # type: ignore[arg-type]
            k=k,
            rrf_k=rrf_k,
        )

    # --- Graph modes: needs graph_store ---------------------------------------
    if retrieval_mode in ("graph", "hybrid_graph"):
        if graph_store is None:
            logger.warning(
                "[build_retriever] retrieval_mode=%r requires graph_store but it is None. "
                "Falling back to 'vector'.",
                retrieval_mode,
            )
            return vector_retriever

        graph_retriever = GraphRetriever(
            graph_store=graph_store,
            k=k,
            max_depth=2,
            category=category,
        )

        if retrieval_mode == "graph":
            logger.info("[build_retriever] mode=graph (category=%r k=%d depth=2)", category, k)
            return graph_retriever

        if retrieval_mode == "hybrid_graph":
            logger.info(
                "[build_retriever] mode=hybrid_graph (category=%r k=%d rrf_k=%d)",
                category, k, rrf_k,
            )
            return HybridRetriever(
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,  # type: ignore[arg-type]
                graph_retriever=graph_retriever,
                k=k,
                rrf_k=rrf_k,
            )

    logger.warning(
        "[build_retriever] unknown retrieval_mode=%r — falling back to 'vector'.",
        retrieval_mode,
    )
    return vector_retriever
