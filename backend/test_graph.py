"""
Smoke test for all 5 retrieval modes.
Run from the backend/ directory: python test_graph.py
"""
import asyncio
import sys
sys.path.insert(0, '.')

from app.rag.embeddings import get_embeddings
from app.rag.vector_store import get_vector_store
from app.rag.knowledge_graph import get_knowledge_graph
from app.rag.graph import get_graph_store
from app.rag.retriever import build_retriever

QUERY = "What is the difference in temperature reading accuracy between DHT11 and DHT22?"

async def test_mode(mode: str, embeddings, vector_store, kg, gs, k=4):
    print(f"\n{'='*58}")
    print(f"MODE: {mode.upper()}")
    print(f"{'='*58}")
    retriever = build_retriever(
        embeddings, vector_store, k,
        category=None,
        knowledge_graph=kg,
        retrieval_mode=mode,
        graph_store=gs,
    )
    docs = await retriever.ainvoke(QUERY)
    print(f"  Returned {len(docs)} document(s):")
    for i, d in enumerate(docs, 1):
        via  = d.metadata.get("retrieved_via", "?")
        rrf  = d.metadata.get("rrf_score")
        sim  = d.metadata.get("similarity")
        bm   = d.metadata.get("bm25_score")
        grph = d.metadata.get("graph_score")
        score_str = (
            f"rrf={rrf:.5f}" if rrf else
            f"sim={sim:.4f}" if sim else
            f"bm25={bm:.4f}" if bm else
            f"graph={grph:.4f}" if grph else
            "no-score"
        )
        print(f"  [{i}] via={via:<20} {score_str}  | {d.page_content[:55]}...")

async def main():
    embeddings = get_embeddings()
    vector_store = get_vector_store()
    kg = get_knowledge_graph()
    gs = get_graph_store()
    print(f"KnowledgeGraph : {kg.node_count} nodes, {kg.edge_count} edges")
    print(f"GraphStore     : {gs._graph.number_of_nodes()} entity nodes, {gs._graph.number_of_edges()} edges")
    print(f"Alias index    : {len(gs._alias_index)} aliases")
    print(f"\nQuery: {QUERY!r}")

    for mode in ("vector", "bm25", "hybrid", "graph", "hybrid_graph"):
        await test_mode(mode, embeddings, vector_store, kg, gs)

    print("\n\nAll modes completed successfully.")

if __name__ == "__main__":
    asyncio.run(main())
