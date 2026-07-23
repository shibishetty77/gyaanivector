"""
Vector store access.

This does NOT create a vector store — it talks to the `knowledge_chunks`
table and `match_knowledge_chunks` RPC function that already exist in your
Supabase project (see the ingestion pipeline's schema.sql). This module's
only job is: given a query embedding, call that RPC and return raw rows.

The official supabase-py client is synchronous under the hood, so calls are
offloaded to a thread via `asyncio.to_thread` to keep the FastAPI event loop
free. This is the pragmatic, stable choice — supabase-py's native async
client support has shifted across versions, while this approach works with
any 2.x release and is easy to swap out later if you upgrade.
"""

import asyncio
import logging
from functools import lru_cache
from typing import Any

from supabase import Client, create_client

from app.config import get_settings

logger = logging.getLogger(__name__)


class SupabaseVectorStore:
    """Read-only accessor for the existing knowledge_chunks table / RPC function."""

    def __init__(self, client: Client, match_function: str) -> None:
        self._client = client
        self._match_function = match_function

    def similarity_search(
        self,
        query_embedding: list[float],
        k: int,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Synchronous call to the existing match_knowledge_chunks RPC function."""
        logger.info(
            "[STAGE: Supabase RPC] entered stage | rpc=%s embedding_length=%d category=%r k=%d",
            self._match_function, len(query_embedding), category, k,
        )
        params = {
            "query_embedding": query_embedding,
            "match_count": k,
            "filter_category": category,
        }
        try:
            response = self._client.rpc(self._match_function, params).execute()
        except Exception as exc:
            logger.error(
                "[STAGE: Supabase RPC] failure reason | rpc=%s error=%s: %s",
                self._match_function, type(exc).__name__, exc,
            )
            raise
        rows = response.data or []
        doc_ids = [row.get("doc_id") or row.get("source_path") for row in rows]
        similarities = [row.get("similarity") for row in rows]
        logger.info(
            "[STAGE: Supabase RPC] completed stage | rows_returned=%d doc_ids=%s similarities=%s",
            len(rows), doc_ids, similarities,
        )
        return rows

    async def asimilarity_search(
        self,
        query_embedding: list[float],
        k: int,
        category: str | None = None,
    ) -> list[dict[str, Any]]:
        """Async wrapper — see module docstring for why this is thread-offloaded."""
        return await asyncio.to_thread(self.similarity_search, query_embedding, k, category)


@lru_cache
def get_vector_store() -> SupabaseVectorStore:
    settings = get_settings()
    client = create_client(settings.supabase_url, settings.supabase_service_key)
    logger.info("Connected to Supabase project for vector search (table=%s).", settings.knowledge_table)
    return SupabaseVectorStore(client=client, match_function=settings.match_function)
