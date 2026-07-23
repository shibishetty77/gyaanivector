"""
API routes.

POST /chat          -> full answer + sources (JSON)
POST /chat/stream    -> Server-Sent-Events style token stream, sources sent first
"""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag.chain import RAGService

logger = logging.getLogger(__name__)
router = APIRouter()


class Source(BaseModel):
    source_path: str | None = None
    category: str | None = None
    similarity: float | None = None
    retrieved_via: str = "vector"  # "vector" or "graph" (see knowledge_graph.py)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    category: str | None = Field(
        None, description="Optional metadata filter, e.g. 'facts', 'classes', 'eval'"
    )
    k: int | None = Field(None, ge=1, le=20, description="Number of documents to retrieve")
    session_id: str | None = Field(
        None, description="Optional session id to enable multi-turn conversation memory"
    )


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source]


def get_rag_service(request: Request) -> RAGService:
    """RAGService is built once at startup and stored on app.state (see main.py)."""
    service: RAGService | None = getattr(request.app.state, "rag_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="RAG service is not initialized yet.")
    return service


@router.post("/chat", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> ChatResponse:
    logger.info(
        "[STAGE: API] entered stage | question_preview=%r category=%r k=%r session_id=%r",
        payload.question[:80], payload.category, payload.k, payload.session_id,
    )
    try:
        result = await rag_service.answer(
            question=payload.question,
            category=payload.category,
            k=payload.k,
            session_id=payload.session_id,
        )
    except Exception:
        logger.error("[STAGE: API] failure reason | unhandled exception during rag_service.answer", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.")

    logger.info(
        "[STAGE: API] completed stage | answer_length=%d num_sources=%d",
        len(result["answer"]), len(result["sources"]),
    )
    return ChatResponse(answer=result["answer"], sources=result["sources"])


@router.post("/chat/stream")
async def chat_stream(
    payload: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> StreamingResponse:
    try:
        documents, context = await rag_service.retrieve_with_context(
            question=payload.question, category=payload.category, k=payload.k
        )
    except Exception:
        logger.exception("Error retrieving documents")
        raise HTTPException(status_code=500, detail="Failed to retrieve context.")

    sources = rag_service.to_sources(documents)

    async def event_generator():
        # First event carries sources, so the client can render them immediately
        # without waiting for generation to finish.
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"
        try:
            async for token in rag_service.stream_answer(
                question=payload.question, context=context, session_id=payload.session_id
            ):
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception:
            logger.exception("Error during streaming generation")
            yield f"event: error\ndata: {json.dumps({'detail': 'Generation failed.'})}\n\n"
            return
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
