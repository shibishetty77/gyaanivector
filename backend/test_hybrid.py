"""
Quick smoke test for BM25 and Hybrid retriever modes.
Run from the backend/ directory: python test_hybrid.py
Requires SUPABASE_URL, SUPABASE_SERVICE_KEY, GOOGLE_API_KEY in .env
"""
import asyncio
import sys
import os
sys.path.insert(0, '.')

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import get_vector_store
from app.rag.knowledge_graph import get_knowledge_graph
from app.rag.retriever import build_retriever

QUERY = "What is the difference in temperature reading accuracy between DHT11 and DHT22?"

async def test_mode(mode: str, embeddings, vector_store, kg, k=4):
    print(f"\n{'='*55}")
    print(f"MODE: {mode.upper()}")
    print(f"{'='*55}")
    retriever = build_retriever(
        embeddings, vector_store, k,
        category=None,
        knowledge_graph=kg,
        retrieval_mode=mode,
    )
    docs = await retriever.ainvoke(QUERY)
    print(f"  Returned {len(docs)} document(s):")
    for i, d in enumerate(docs, 1):
        via = d.metadata.get("retrieved_via", "?")
        rrf = d.metadata.get("rrf_score")
        sim = d.metadata.get("similarity")
        bm25 = d.metadata.get("bm25_score")
        score_str = (f"rrf={rrf:.5f}" if rrf else
                     f"sim={sim:.4f}" if sim else
                     f"bm25={bm25:.4f}" if bm25 else "no-score")
        print(f"  [{i}] via={via:<14} {score_str}  | {d.page_content[:60]}...")

async def main():
    settings = get_settings()
    embeddings = get_embeddings()
    vector_store = get_vector_store()
    kg = get_knowledge_graph()
    print(f"Knowledge graph: {kg.node_count} nodes, {kg.edge_count} edges")
    print(f"Query: {QUERY!r}")

    for mode in ("vector", "bm25", "hybrid"):
        await test_mode(mode, embeddings, vector_store, kg)

    print("\n\nAll modes completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
