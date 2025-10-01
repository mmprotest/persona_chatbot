"""Application configuration management."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(slots=True)
class LLMConfig:
    """Configuration for an LLM backend."""

    provider: str = os.getenv("LLM_PROVIDER", "openai").lower()
    model: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    api_key: Optional[str] = os.getenv("LLM_API_KEY")
    base_url: Optional[str] = os.getenv("LLM_BASE_URL")
    request_timeout: float = float(os.getenv("LLM_TIMEOUT", "120"))


@dataclass(slots=True)
class MemoryConfig:
    """Configuration for the long-term memory store."""

    database_path: str = os.getenv("MEMORY_DB_PATH", "./data/memory.sqlite")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "384"))
    max_results: int = int(os.getenv("MEMORY_MAX_RESULTS", "6"))
    relevance_threshold: float = float(os.getenv("MEMORY_RELEVANCE_THRESHOLD", "0.35"))


@dataclass(slots=True)
class PersonaConfig:
    """Persona configuration options."""

    name: str = os.getenv("PERSONA_NAME", "Avery")
    description: str = os.getenv("PERSONA_DESCRIPTION", "A thoughtful AI companion who is empathetic, curious, and articulate.")
    goals: str = os.getenv(
        "PERSONA_GOALS",
        (
            "Maintain engaging, realistic conversations; adapt to the user's mood and intent; "
            "provide insightful, context-aware responses; and reflect on prior interactions to "
            "grow the relationship over time."
        ),
    )
    seed_prompt: str = os.getenv(
        "PERSONA_SEED",
        "",
    )


@dataclass(slots=True)
class AppConfig:
    """Top-level application configuration."""

    llm: LLMConfig = LLMConfig()
    memory: MemoryConfig = MemoryConfig()
    persona: PersonaConfig = PersonaConfig()


config = AppConfig()
