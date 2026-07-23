"""
Graph visualization API routes.

GET  /graph          → serves the interactive HTML graph visualizer
GET  /graph/data     → returns the graph as JSON (nodes + edges)
GET  /graph/query    → highlight nodes matched by a query string
"""

import logging
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from app.rag.graph.graph_store import GraphStore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/graph", tags=["graph"])

_VIEWER_PATH = Path(__file__).parent.parent.parent / "graph_viewer.html"


def _get_graph_store(request: Request) -> GraphStore | None:
    return getattr(request.app.state, "graph_store", None)


# ---------------------------------------------------------------------------
# /graph/data — raw graph JSON for any frontend / external tool
# ---------------------------------------------------------------------------

@router.get("/data")
async def graph_data(request: Request) -> JSONResponse:
    """Return the knowledge graph as a JSON object with nodes and edges arrays."""
    gs = _get_graph_store(request)
    if gs is None:
        return JSONResponse({"nodes": [], "edges": [], "error": "Graph store not initialized"}, status_code=503)

    g = gs._graph
    docs_by_id = gs._docs_by_id

    # Assign a colour bucket per relation type
    CATEGORY_COLORS = {
        "facts": "#6EE7B7",     # green
        "classes": "#93C5FD",   # blue
        "eval": "#FCA5A5",      # red
        "interface": "#FDE68A", # yellow
        "entity": "#C4B5FD",    # purple
    }

    def node_category(node_id: str) -> str:
        """Figure out a display category for a node."""
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

        # Build a short label (prefer the node_id itself, max 20 chars)
        label = node_id[:20] if len(node_id) <= 20 else node_id[:17] + "…"

        # Gather source paths for tooltip
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

    return JSONResponse({
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "node_count": g.number_of_nodes(),
            "edge_count": g.number_of_edges(),
        },
    })


# ---------------------------------------------------------------------------
# /graph/query — which nodes does a query activate?
# ---------------------------------------------------------------------------

@router.get("/query")
async def graph_query(request: Request, q: str = "") -> JSONResponse:
    """Return the set of entity IDs matched by the query (for client-side highlighting)."""
    gs = _get_graph_store(request)
    if gs is None:
        return JSONResponse({"matched": [], "traversed": {}}, status_code=503)

    entities = gs.extract_entities_from_query(q) if q else set()
    traversed = gs.traverse(entities, max_depth=2)

    return JSONResponse({
        "query": q,
        "matched_entities": list(entities),
        "traversed": {k: round(v, 4) for k, v in traversed.items()},
    })


# ---------------------------------------------------------------------------
# /graph — serve the interactive HTML visualizer
# ---------------------------------------------------------------------------

@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def graph_viewer() -> HTMLResponse:
    """Serve the interactive D3 force-directed graph viewer."""
    if _VIEWER_PATH.exists():
        return HTMLResponse(_VIEWER_PATH.read_text(encoding="utf-8"))
    return HTMLResponse("<h2>graph_viewer.html not found next to this package.</h2>", status_code=404)
