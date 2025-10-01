"""Ollama chat completion client."""
from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional

import requests

from .base import LLMClient


class OllamaClient(LLMClient):
    """Client for the Ollama local inference server."""

    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        request_timeout: float = 120.0,
    ) -> None:
        super().__init__(model=model, request_timeout=request_timeout)
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")

    def complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        response = requests.post(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload),
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message", {})
        content = message.get("content")
        if not content:
            raise RuntimeError("Ollama response did not include content")
        return content.strip()

    def stream_complete(
        self,
        messages: Iterable[dict[str, str]],
        *,
        max_tokens: Optional[int] = None,
    ) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": True,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}
        with requests.post(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload),
            timeout=self.request_timeout,
            stream=True,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                message = data.get("message", {})
                content = message.get("content")
                if content:
                    yield content
