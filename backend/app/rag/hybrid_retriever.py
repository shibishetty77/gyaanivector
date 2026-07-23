"""
Reciprocal Rank Fusion (RRF) and the HybridRetriever that combines dense
(pgvector) and sparse (BM25) retrieval results.

RRF formula (Cormack et al., 2009):
    score(d) = Σ  1 / (k + rank_i(d))
               i
where k=60 is the standard constant and rank_i(d) is the 1-based rank of
document d in result list i.  Documents not in a list get no contribution
from that list.  Higher score = better.

Why RRF here instead of a learned re-ranker:
  - Zero additional model or API calls — purely arithmetic.
  - Proven to be highly competitive with more complex fusion methods on
    standard retrieval benchmarks.
  - k=60 makes high-ranked documents dominate while still giving tail docs
    a non-zero contribution, avoiding over-weighting any single list.

HybridRetriever fires both retrievers concurrently (asyncio.gather) then
fuses their ranked lists.  The final result is deduplicated by doc_id and
truncated to `k` documents, preserving the `retrieved_via` tag from each
document's highest-ranking source.
"""

import asyncio
import logging
from collections import defaultdict
from typing import Any

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pure RRF fusion function — no dependencies, easily unit-testable
# ---------------------------------------------------------------------------

def reciprocal_rank_fusion(
    ranked_lists: list[list[Document]],
    rrf_k: int = 60,
) -> list[Document]:
    """
    Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    Args:
        ranked_lists: Each inner list is an ordered list of Documents from
                      one retriever (best first).
        rrf_k:        The RRF constant (60 per the original paper).

    Returns:
        A new list of Documents sorted by descending RRF score, deduplicated
        by ``doc_id`` (falling back to ``source_path`` if doc_id is absent).
        The returned Document objects carry their RRF score in
        ``metadata["rrf_score"]``.
    """
    # Map doc_key → aggregated RRF score
    rrf_scores: dict[str, float] = defaultdict(float)
    # Map doc_key → best Document object seen (preserves metadata)
    best_doc: dict[str, Document] = {}
    # Map doc_key → set of retrieval origins for `retrieved_via`
    origins: dict[str, set[str]] = defaultdict(set)

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.metadata.get("doc_id") or doc.metadata.get("source_path") or doc.page_content[:64]
            rrf_scores[key] += 1.0 / (rrf_k + rank)
            origins[key].add(doc.metadata.get("retrieved_via", "unknown"))
            # Keep the first (highest-ranked) version of each document
            if key not in best_doc:
                best_doc[key] = doc

    # Build final sorted list
    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    result: list[Document] = []
    for key in sorted_keys:
        doc = best_doc[key]
        meta = dict(doc.metadata)
        meta["rrf_score"] = round(rrf_scores[key], 6)
        meta["retrieved_via"] = "+".join(sorted(origins[key]))  # e.g. "bm25+vector"
        result.append(Document(page_content=doc.page_content, metadata=meta))

    return result


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """
    Combines a dense (vector) retriever and a sparse (BM25) retriever via RRF.

    Both retrievers are invoked concurrently.  Their ranked lists are fused
    with Reciprocal Rank Fusion and the top-k results are returned.

    Metadata on each returned Document:
      - ``retrieved_via``: comma-joined set of retrieval paths that surfaced it,
                           e.g. ``"vector"`` / ``"bm25"`` / ``"bm25+vector"``
      - ``rrf_score``:     the fused RRF score (higher is better)
      - ``similarity`` / ``bm25_score``: preserved from whichever retriever
                                         contributed the stored Document copy
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    k: int = 5
    rrf_k: int = 60

    # Non-serializable retrievers stored as private attrs
    _vector_retriever: Any
    _bm25_retriever: Any
    _graph_retriever: Any | None

    def __init__(
        self,
        vector_retriever: BaseRetriever,
        bm25_retriever: BaseRetriever,
        graph_retriever: BaseRetriever | None = None,
        k: int = 5,
        rrf_k: int = 60,
    ) -> None:
        super().__init__(k=k, rrf_k=rrf_k)
        self._vector_retriever = vector_retriever
        self._bm25_retriever = bm25_retriever
        self._graph_retriever = graph_retriever

    def _fuse(self, doc_lists: list[list[Document]]) -> list[Document]:
        fused = reciprocal_rank_fusion(doc_lists, rrf_k=self.rrf_k)
        top_k = fused[: self.k]
        lengths = [len(lst) for lst in doc_lists]
        logger.info(
            "[HybridRetriever] RRF fusion: lists=%s → fused=%d → top_k=%d",
            lengths, len(fused), len(top_k),
        )
        return top_k

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Sync path
        doc_lists = [
            self._vector_retriever.invoke(query),
            self._bm25_retriever.invoke(query),
        ]
        if self._graph_retriever:
            doc_lists.append(self._graph_retriever.invoke(query))
            
        return self._fuse(doc_lists)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # Async path: fire all concurrently
        tasks = [
            self._vector_retriever.ainvoke(query),
            self._bm25_retriever.ainvoke(query),
        ]
        if self._graph_retriever:
            tasks.append(self._graph_retriever.ainvoke(query))
            
        results = await asyncio.gather(*tasks)
        return self._fuse(list(results))
