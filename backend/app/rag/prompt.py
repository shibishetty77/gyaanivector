"""
Prompt construction for the RAG chain.

Kept in its own module so the persona / instructions can be tuned without
touching retrieval or chain-wiring logic.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_PROMPT = """You are Gyani AI.

You are an embedded systems expert.

Answer ONLY using the retrieved context.

If the answer cannot be found in the retrieved documents, clearly say you don't know.

Never hallucinate hardware specifications.

Always prefer datasheet information.

Refer to the database for hardware related things, and also mention the supporting components and devices you can explain via the knowledge files.

Also refer yo the datasheets and pin configurations.

If asked a general "what is X" question, you may synthesize a short description
purely from the specific facts retrieved for X (e.g. its class/category, interface
type, pin counts, voltage) — but never state a fact that isn't present in the context.

Context:
{context}"""


def build_prompt_template() -> ChatPromptTemplate:
    """
    Chat-style prompt with:
      - a system turn carrying the persona + retrieved context
      - an optional chat history placeholder (used when conversation memory is enabled)
      - the current human question
    """
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{question}"),
        ]
    )


def format_docs(documents: list[Document]) -> str:
    """Render retrieved Documents into the {context} block the prompt expects."""
    if not documents:
        return "(no relevant documents were retrieved)"

    blocks = []
    for doc in documents:
        source = doc.metadata.get("source_path", "unknown source")
        blocks.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(blocks)