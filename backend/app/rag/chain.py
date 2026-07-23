"""
Wires retrieval -> prompt -> LLM into the actual RAG chain, using modern
LCEL (LangChain Expression Language) -- RunnableSequence via `|`, plus
RunnableLambda for the context-formatting step and StrOutputParser for
output parsing. No deprecated `LLMChain`/`RetrievalQA` APIs.

Conversation memory is implemented as a plain in-process
`dict[session_id, list[BaseMessage]]` rather than LangChain's
`RunnableWithMessageHistory` (deprecated upstream in favor of LangGraph
persistence as of recent LangChain versions). This is deliberately the
simplest thing that works: a list of messages per session, read before the
call and appended to after. It also maps directly onto LangGraph's own
model (state = message list, persisted by a checkpointer), so migrating
later is a drop-in swap of the storage layer, not a rewrite of this logic.

Kept as a single `RAGService` class (rather than one giant module-level
chain) so retrieval, source extraction, and generation stay independently
testable, and each method here maps cleanly onto a future LangGraph node.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from app.rag.embeddings import LocalQueryEmbeddings
from app.rag.knowledge_graph import KnowledgeGraph
from app.rag.prompt import format_docs
from app.rag.retriever import build_retriever
from app.rag.vector_store import SupabaseVectorStore

logger = logging.getLogger(__name__)

# --- Conversation memory --------------------------------------------------
# In-memory, per-process session store. Fine for a single-instance deployment
# / development. For persistence across restarts or multiple app instances,
# swap this dict for a Supabase-backed store (e.g. a `chat_sessions` table
# keyed by session_id) -- nothing else in RAGService needs to change, since
# access is already isolated behind _load_history / _save_turn below.
_session_store: dict[str, list[BaseMessage]] = {}


def _load_history(session_id: str) -> list[BaseMessage]:
    return _session_store.setdefault(session_id, [])


def _save_turn(session_id: str, question: str, answer: str) -> None:
    history = _session_store.setdefault(session_id, [])
    history.append(HumanMessage(content=question))
    history.append(AIMessage(content=answer))


class RAGService:
    """High-level entry point used by the API layer: ask a question, get an answer + sources."""

    def __init__(
        self,
        embeddings: LocalQueryEmbeddings,
        vector_store: SupabaseVectorStore,
        llm: BaseChatModel,
        prompt: ChatPromptTemplate,
        default_k: int = 5,
        knowledge_graph: KnowledgeGraph | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.vector_store = vector_store
        self.default_k = default_k
        self.knowledge_graph = knowledge_graph
        # prompt | llm | parser -- a RunnableSequence, LangChain's modern LCEL
        # replacement for the deprecated LLMChain.
        self.chain = prompt | llm | StrOutputParser()

    async def _retrieve(self, question: str, k: int | None, category: str | None) -> list[Document]:
        retriever = build_retriever(
            self.embeddings, self.vector_store, k or self.default_k, category,
            knowledge_graph=self.knowledge_graph,
        )
        return await retriever.ainvoke(question)

    @staticmethod
    def to_sources(documents: list[Document]) -> list[dict[str, Any]]:
        return [
            {
                "source_path": doc.metadata.get("source_path"),
                "category": doc.metadata.get("category"),
                "similarity": doc.metadata.get("similarity"),
                "retrieved_via": doc.metadata.get("retrieved_via", "vector"),
            }
            for doc in documents
        ]

    async def answer(
        self,
        question: str,
        category: str | None = None,
        k: int | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        documents = await self._retrieve(question, k, category)

        logger.info("[STAGE: Context assembly] entered stage | num_documents=%d", len(documents))
        context = await RunnableLambda(format_docs).ainvoke(documents)
        chunk_previews = [doc.page_content[:200] for doc in documents]
        logger.info(
            "[STAGE: Context assembly] completed stage | num_chunks=%d total_context_chars=%d chunk_previews=%s",
            len(documents), len(context), chunk_previews,
        )

        chat_history = _load_history(session_id) if session_id else []

        logger.info("[STAGE: LLM] started | provider_class=%s", type(self.chain).__name__)
        answer_text = await self.chain.ainvoke(
            {"question": question, "context": context, "chat_history": chat_history}
        )
        logger.info("[STAGE: LLM] finished | response_length=%d", len(answer_text))

        if session_id:
            _save_turn(session_id, question, answer_text)

        logger.info("[STAGE: Final response] completed stage | answer_length=%d num_sources=%d", len(answer_text), len(documents))
        return {"answer": answer_text, "sources": self.to_sources(documents)}

    async def retrieve_with_context(
        self, question: str, category: str | None = None, k: int | None = None
    ) -> tuple[list[Document], str]:
        """Exposed separately so callers (e.g. the streaming endpoint) can grab
        sources up front, before/alongside streaming the generated answer."""
        documents = await self._retrieve(question, k, category)
        context = await RunnableLambda(format_docs).ainvoke(documents)
        return documents, context

    async def stream_answer(
        self,
        question: str,
        context: str,
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """Yields answer tokens as they're generated, given pre-fetched context
        (see `retrieve_with_context`). Appends the full turn to session memory
        once streaming completes."""
        chat_history = _load_history(session_id) if session_id else []
        payload = {"question": question, "context": context, "chat_history": chat_history}

        collected: list[str] = []
        async for chunk in self.chain.astream(payload):
            collected.append(chunk)
            yield chunk

        if session_id:
            _save_turn(session_id, question, "".join(collected))
