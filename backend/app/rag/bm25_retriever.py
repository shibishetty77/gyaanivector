"""
Sparse BM25 retriever backed by the same document corpus loaded into the
KnowledgeGraph at startup.

Why reuse KnowledgeGraph.docs_by_id:
  - The KnowledgeGraph already fetches every row (doc_id, content, metadata)
    from Supabase once at startup — zero extra DB calls needed here.
  - Tokens are indexed once in __init__, then queries are pure in-memory.

BM25 complements pgvector well:
  - Vector search finds *semantically* similar chunks (synonyms, paraphrasing).
  - BM25 finds chunks with strong *lexical* overlap (exact keywords, part
    numbers, pin names, peripheral abbreviations like "I2C", "SPI", "GPIO").
  For a hardware knowledge base this matters: "GPIO34" and "I2C address 0x27"
  are exact tokens that embeddings can under-weight.

This class is intentionally thin — it knows nothing about the chain, prompt,
or LLM. It is only ever called by HybridRetriever or directly by
build_retriever() when RETRIEVAL_MODE=bm25.
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

logger = logging.getLogger(__name__)


def _simple_tokenize(text: str) -> list[str]:
    """Lowercase, split on whitespace/punctuation — fast, no NLTK dependency."""
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Retriever(BaseRetriever):
    """
    LangChain-compatible sparse retriever using BM25Okapi from rank-bm25.

    Corpus is provided at construction time as ``docs_by_id`` — the same dict
    already maintained by KnowledgeGraph.  An optional ``category`` filter
    mirrors the behaviour of SupabaseRAGRetriever so both retrievers can be
    called with identical arguments.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Serializable fields (Pydantic models must declare these)
    k: int = 5
    category: str | None = None

    # Non-serializable state — stored as private attrs, set in __init__
    _corpus_docs: list[Document]
    _bm25: Any  # BM25Okapi instance

    def __init__(
        self,
        docs_by_id: dict[str, dict[str, Any]],
        k: int = 5,
        category: str | None = None,
    ) -> None:
        super().__init__(k=k, category=category)
        # Filter by category if requested (same semantics as vector retriever)
        rows = list(docs_by_id.values())
        if category is not None:
            rows = [r for r in rows if r.get("category") == category]

        self._corpus_docs = [
            Document(
                page_content=row.get("content", ""),
                metadata={
                    "source_path": row.get("source_path"),
                    "category": row.get("category"),
                    "subcategory": row.get("subcategory"),
                    "doc_id": row.get("doc_id"),
                    "similarity": None,       # BM25 scores are not cosine-sim
                    "retrieved_via": "bm25",
                },
            )
            for row in rows
        ]

        tokenized_corpus = [_simple_tokenize(doc.page_content) for doc in self._corpus_docs]

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise ImportError(
                "rank-bm25 is required for BM25 retrieval. "
                "Install it with: pip install rank-bm25"
            ) from exc

        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(
            "[BM25Retriever] indexed %d documents (category=%r)",
            len(self._corpus_docs), category,
        )

    def _query(self, query: str) -> list[Document]:
        tokens = _simple_tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Pair each doc with its BM25 score and sort descending
        scored = sorted(
            zip(scores, self._corpus_docs),
            key=lambda x: x[0],
            reverse=True,
        )

        top_k = scored[: self.k]
        results: list[Document] = []
        for score, doc in top_k:
            # Skip docs with zero BM25 score — they share no query tokens
            if score <= 0.0:
                continue
            # Store the raw BM25 score in metadata for transparency
            enriched_meta = dict(doc.metadata)
            enriched_meta["bm25_score"] = float(score)
            results.append(Document(page_content=doc.page_content, metadata=enriched_meta))

        logger.info(
            "[BM25Retriever] query=%r | corpus=%d | k=%d | returned=%d",
            query[:80], len(self._corpus_docs), self.k, len(results),
        )
        return results

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        return self._query(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: AsyncCallbackManagerForRetrieverRun
    ) -> list[Document]:
        # BM25 scoring is CPU-bound; offload to a thread to keep event loop free
        return await asyncio.to_thread(self._query, query)
