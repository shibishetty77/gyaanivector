"""
FastAPI application entrypoint.

Everything expensive (loading the embedding model, building the LLM client)
happens once at startup and is stashed on `app.state`, not per-request.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router as chat_router
from app.api.graph_routes import router as graph_router
from app.config import get_settings
from app.rag.chain import RAGService
from app.rag.embeddings import get_embeddings
from app.rag.graph import get_graph_store
from app.rag.knowledge_graph import get_knowledge_graph
from app.rag.prompt import build_prompt_template
from app.rag.vector_store import get_vector_store
from app.services.llm import get_llm

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up: loading embedding model, connecting Supabase, initializing LLM...")

    embeddings = get_embeddings()
    vector_store = get_vector_store()
    llm = get_llm()
    prompt = build_prompt_template()
    knowledge_graph = get_knowledge_graph()
    logger.info(
        "Knowledge graph ready: %d nodes, %d edges.",
        knowledge_graph.node_count, knowledge_graph.edge_count,
    )

    graph_store = get_graph_store()
    logger.info(
        "Graph store ready: %d entity nodes, %d edges.",
        graph_store._graph.number_of_nodes(),
        graph_store._graph.number_of_edges(),
    )
    app.state.graph_store = graph_store  # expose to routes

    app.state.rag_service = RAGService(
        embeddings=embeddings,
        vector_store=vector_store,
        llm=llm,
        prompt=prompt,
        default_k=settings.default_match_count,
        knowledge_graph=knowledge_graph,
        retrieval_mode=settings.retrieval_mode,
        rrf_k=settings.rrf_k,
        graph_store=graph_store,
    )
    logger.info("Retrieval mode: %s (rrf_k=%d).", settings.retrieval_mode, settings.rrf_k)

    logger.info("Startup complete.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Gyani AI — RAG API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)
app.include_router(graph_router)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
