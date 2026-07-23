"""
LLM factory.

Everything downstream (chain.py) only depends on the LangChain `Runnable`
interface (`ainvoke`, `astream`, ...), so switching providers is a one-line
env var change (`LLM_PROVIDER`) — no code elsewhere needs to change.
"""

import logging
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


def _build_gemini(settings: Settings) -> BaseChatModel:
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not settings.google_api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set but LLM_PROVIDER=gemini")

    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.google_api_key,
        temperature=settings.llm_temperature,
        convert_system_message_to_human=False,
    )


def _build_openai(settings: Settings) -> BaseChatModel:
    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not set but LLM_PROVIDER=openai")

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openai_api_key,
        temperature=settings.llm_temperature,
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set but LLM_PROVIDER=anthropic")

    return ChatAnthropic(
        model=settings.llm_model,
        api_key=settings.anthropic_api_key,
        temperature=settings.llm_temperature,
    )


def _build_ollama(settings: Settings) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    # OLLAMA_BASE_URL is optional; fall back to Ollama's default local
    # endpoint if it's unset or set to an empty string.
    base_url = settings.ollama_base_url or "http://localhost:11434"

    return ChatOllama(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        base_url=base_url,
    )


def _build_openrouter(settings: Settings) -> BaseChatModel:
    # OpenRouter exposes an OpenAI-compatible API, so the standard OpenAI
    # integration works unmodified — only the base_url and key differ.
    from langchain_openai import ChatOpenAI

    if not settings.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set but LLM_PROVIDER=openrouter")

    base_url = settings.openrouter_base_url or "https://openrouter.ai/api/v1"

    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=base_url,
        temperature=settings.llm_temperature,
    )


_PROVIDERS = {
    "gemini": _build_gemini,
    "openai": _build_openai,
    "anthropic": _build_anthropic,
    "ollama": _build_ollama,
    "openrouter": _build_openrouter,
}


@lru_cache
def get_llm() -> BaseChatModel:
    """Built once per process. Provider + model come entirely from env config."""
    settings = get_settings()
    builder = _PROVIDERS.get(settings.llm_provider)
    if builder is None:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.llm_provider}'. "
            f"Valid options: {', '.join(_PROVIDERS)}"
        )
    logger.info("Initializing LLM provider=%s model=%s", settings.llm_provider, settings.llm_model)
    return builder(settings)
