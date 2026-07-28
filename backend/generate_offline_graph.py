import json
import logging
import asyncio
from pathlib import Path
import sys

sys.path.insert(0, '.')

from app.rag.embeddings import get_embeddings
from app.rag.vector_store import get_vector_store
from app.rag.knowledge_graph import get_knowledge_graph
from app.rag.graph.builder import get_graph_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_offline_html():
    logger.info("Initializing components...")
    kg = get_knowledge_graph()
    gs = get_graph_store()
    
    logger.info("Extracting graph data...")
    g = gs._graph
    docs_by_id = gs._docs_by_id

    CATEGORY_COLORS = {
        "facts": "#6EE7B7",
        "classes": "#93C5FD",
        "eval": "#FCA5A5",
        "interface": "#FDE68A",
        "entity": "#C4B5FD",
    }

    def node_category(node_id: str) -> str:
        data = g.nodes[node_id]
        doc_ids = data.get("doc_ids", set())
        for doc_id in doc_ids:
            row = docs_by_id.get(doc_id)
            if row:
                return row.get("category") or "entity"
        return "entity"

    nodes = []
    for node_id, data in g.nodes(data=True):
        cat = node_category(node_id)
        doc_ids = list(data.get("doc_ids", set()))
        aliases = list(data.get("aliases", set()))

        label = node_id[:20] if len(node_id) <= 20 else node_id[:17] + "…"

        sources = []
        for did in doc_ids:
            row = docs_by_id.get(did)
            if row:
                sp = row.get("source_path", "")
                if sp:
                    sources.append(sp)

        nodes.append({
            "id": node_id,
            "label": label,
            "category": cat,
            "color": CATEGORY_COLORS.get(cat, "#E2E8F0"),
            "aliases": aliases,
            "doc_ids": doc_ids,
            "sources": sources,
        })

    edges = []
    for src, dst, data in g.edges(data=True):
        relation = data.get("relation", "related")
        edges.append({
            "source": src,
            "target": dst,
            "relation": relation,
            "label": relation.replace("_", " "),
        })

    graph_data = {
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": g.number_of_nodes(),
            "edge_count": g.number_of_edges(),
        }
    }
    
    # Read the existing HTML
    html_path = Path("graph_viewer.html")
    if not html_path.exists():
        logger.error("graph_viewer.html not found.")
        return
        
    html_content = html_path.read_text(encoding="utf-8")
    
    # Inject the data
    json_data = json.dumps(graph_data)
    
    # Replace the fetch logic in init()
    old_init = """async function init() {
  try {
    const res = await fetch(GRAPH_DATA_URL);
    graphData = await res.json();
  } catch(e) {
    document.getElementById('loading').innerHTML =
      `<span style="color:var(--rose)">Failed to load graph: ${e.message}</span>`;
    return;
  }"""

    new_init = f"""async function init() {{
  graphData = {json_data};
"""
    
    html_content = html_content.replace(old_init, new_init)
    
    # Replace the fetchRetrievalMode logic
    old_mode = """async function fetchRetrievalMode() {
  try {
    const r = await fetch('/health');
    document.getElementById('stat-mode').textContent = 'hybrid_graph';
  } catch { document.getElementById('stat-mode').textContent = '—'; }
}"""

    new_mode = """async function fetchRetrievalMode() {
  document.getElementById('stat-mode').textContent = 'hybrid_graph (offline)';
}"""
    html_content = html_content.replace(old_mode, new_mode)
    
    # Make the "Query" button fall back to local search since we don't have backend traversal
    old_query = """    // Server-side graph traversal
    try {
      const res = await fetch(`${GRAPH_QUERY_URL}?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      highlightedNodes = new Set(data.matched_entities);
      traversedNodes = data.traversed;
      applyHighlight();
    } catch(e) { console.error(e); }"""
    
    new_query = """    // Fallback to local search for offline mode
    const ql = q.toLowerCase();
    const matched = new Set(graphData.nodes
      .filter(n => n.id.includes(ql) || n.aliases.some(a => a.includes(ql)))
      .map(n => n.id));
    highlightedNodes = matched;
    traversedNodes = {};
    applyHighlight();"""
    html_content = html_content.replace(old_query, new_query)

    # Save to a new file
    output_path = Path("graph_viewer_offline.html")
    output_path.write_text(html_content, encoding="utf-8")
    logger.info(f"Offline HTML generated at: {output_path.absolute()}")

if __name__ == "__main__":
    # Just running the logic synchronously
    generate_offline_html()
