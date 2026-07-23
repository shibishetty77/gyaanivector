"""
Central application configuration.

All settings are loaded from environment variables (or a .env file in the
`backend/` directory). Nothing here talks to Supabase or an LLM directly —
it's just typed, validated config that the rest of the app imports.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Supabase (existing project, existing table — never recreated here) ---
    supabase_url: str = Field(..., alias="SUPABASE_URL")
    supabase_service_key: str = Field(..., alias="SUPABASE_SERVICE_KEY")
    knowledge_table: str = Field("knowledge_chunks", alias="KNOWLEDGE_TABLE")
    match_function: str = Field("match_knowledge_chunks", alias="MATCH_FUNCTION")

    # --- Embeddings (query-time only — ingestion already embedded the docs) ---
    embedding_model: str = Field("all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    embedding_dimensions: int = Field(384, alias="EMBEDDING_DIMENSIONS")

    # --- Retrieval ---
    default_match_count: int = Field(5, alias="DEFAULT_MATCH_COUNT")
    retrieval_mode: str = Field("vector", alias="RETRIEVAL_MODE")
    # "vector" — dense pgvector only (default, backward-compatible)
    # "bm25"   — sparse BM25 only
    # "hybrid" — RRF fusion of vector + BM25
    # "graph"  — entity extraction and graph traversal
    # "hybrid_graph" — RRF fusion of vector + BM25 + graph
    rrf_k: int = Field(60, alias="RRF_K")
    # Reciprocal Rank Fusion constant (default 60 per the original RRF paper)

    # --- LLM (swappable; Gemini 2.5 Flash by default) ---
    llm_provider: str = Field(
        "gemini", alias="LLM_PROVIDER"
    )  # "gemini" | "openai" | "anthropic" | "ollama" | "openrouter"
    llm_model: str = Field("gemini-2.5-flash", alias="LLM_MODEL")
    llm_temperature: float = Field(0.0, alias="LLM_TEMPERATURE")
    google_api_key: str | None = Field(None, alias="GOOGLE_API_KEY")
    openai_api_key: str | None = Field(None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(None, alias="ANTHROPIC_API_KEY")
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    openrouter_api_key: str | None = Field(None, alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field("https://openrouter.ai/api/v1", alias="OPENROUTER_BASE_URL")

    # --- App ---
    cors_origins: list[str] = Field(default_factory=lambda: ["*"], alias="CORS_ORIGINS")
    log_level: str = Field("INFO", alias="LOG_LEVEL")


@lru_cache
def get_settings() -> Settings:
    """Settings are read once and cached — avoids re-parsing env on every request."""
    return Settings()
