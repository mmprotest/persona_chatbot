"""OpenAI-compatible chat completion client."""
from __future__ import annotations

from typing import Iterable, Iterator, Optional

from openai import OpenAI

from .base import LLMClient


class OpenAIClient(LLMClient):
    """Client for OpenAI or OpenAI-compatible APIs."""

    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        request_timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, request_timeout=request_timeout)
        kwargs: dict[str, str] = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens,
            timeout=self.request_timeout,
        )
        if not response.choices:
            raise RuntimeError("LLM returned no choices")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("LLM response had empty content")
        return content.strip()

    def stream_complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=list(messages),
            max_tokens=max_tokens,
            timeout=self.request_timeout,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta.content:
                yield delta.content
