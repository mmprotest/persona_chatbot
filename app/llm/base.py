"""Base interfaces for LLM providers."""
from __future__ import annotations

import abc
from typing import Iterable, List, Optional


class LLMClient(abc.ABC):
    """Abstract base class for an LLM client."""

    def __init__(self, model: str, request_timeout: float = 120.0) -> None:
        self.model = model
        self.request_timeout = request_timeout

    @abc.abstractmethod
    def complete(self, messages: Iterable[dict[str, str]], *, max_tokens: Optional[int] = None) -> str:
        """Generate a chat completion from the provided messages."""

    def reflect(self, prompt: str, *, max_tokens: Optional[int] = None) -> str:
        """Optionally provide a separate reflection call.

        Subclasses can override to provide optimized implementations. By default,
        the method reuses :meth:`complete` with a simple system prompt.
        """

        messages: List[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a reflection and planning module. Provide a concise, actionable "
                    "analysis that helps the assistant improve its next response."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        return self.complete(messages, max_tokens=max_tokens)
