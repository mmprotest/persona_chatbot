"""Factory for instantiating LLM clients based on configuration."""
from __future__ import annotations

from typing import Literal

from ..config import config
from .base import LLMClient
from .ollama_client import OllamaClient
from .openai_client import OpenAIClient

Provider = Literal["openai", "ollama"]


def create_llm_client() -> LLMClient:
    """Create an :class:`LLMClient` based on the current configuration."""

    provider: Provider = config.llm.provider  # type: ignore[assignment]
    if provider == "openai":
        return OpenAIClient(
            model=config.llm.model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
            request_timeout=config.llm.request_timeout,
        )
    if provider == "ollama":
        return OllamaClient(
            model=config.llm.model,
            base_url=config.llm.base_url,
            request_timeout=config.llm.request_timeout,
        )
    raise ValueError(f"Unsupported LLM provider: {provider}")
