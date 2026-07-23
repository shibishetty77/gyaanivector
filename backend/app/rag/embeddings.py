"""
Query-time embeddings.

IMPORTANT: this is only ever used to embed the user's *question*. The
documents in `knowledge_chunks` are already embedded by the existing
ingestion pipeline — we never re-embed or touch document rows here.

Wrapped as a LangChain `Embeddings` implementation so it plugs into any
LangChain component that expects one (retrievers, other vector stores,
future LangGraph nodes, etc.), even though our custom retriever calls
`embed_query` directly most of the time.
"""

import asyncio
import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from app.config import get_settings

logger = logging.getLogger(__name__)


class LocalQueryEmbeddings(Embeddings):
    """Thin LangChain-compatible wrapper around a local SentenceTransformer model."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading local embedding model '%s'...", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded.")

    # --- sync (LangChain's Embeddings ABC requires these) ---
    def embed_query(self, text: str) -> list[float]:
        logger.info("[STAGE: Embedding] entered stage | text_preview=%r", text[:80])
        vector = self._model.encode(text, normalize_embeddings=True)
        result = vector.tolist()
        logger.info("[STAGE: Embedding] completed stage | embedding_length=%d", len(result))
        return result

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Not used in normal operation: documents are embedded once at ingestion
        # time by the existing pipeline, not here. Implemented for interface
        # completeness / potential future reuse (e.g. an offline eval script).
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [v.tolist() for v in vectors]

    # --- async wrappers (SentenceTransformer itself is sync/CPU-bound; we
    # offload to a thread so the FastAPI event loop is never blocked) ---
    async def aembed_query(self, text: str) -> list[float]:
        logger.info("[STAGE: Embedding] entered stage (async) | text_preview=%r", text[:80])
        result = await asyncio.to_thread(self.embed_query, text)
        logger.info("[STAGE: Embedding] completed stage (async) | embedding_length=%d", len(result))
        return result

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await asyncio.to_thread(self.embed_documents, texts)


@lru_cache
def get_embeddings() -> LocalQueryEmbeddings:
    """Loaded once per process and reused — model load is the expensive part."""
    settings = get_settings()
    return LocalQueryEmbeddings(settings.embedding_model)
