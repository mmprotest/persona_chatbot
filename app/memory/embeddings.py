"""Utilities for generating embeddings for long-term memory."""
from __future__ import annotations

import functools
import logging
from typing import Iterable, List

import numpy as np

from ..config import config

_LOGGER = logging.getLogger(__name__)


class EmbeddingService:
    """Generate dense vector embeddings for text snippets."""

    def __init__(self) -> None:
        self._model = None
        self._client = None

    def _ensure_model(self) -> None:
        if self._model is not None or self._client is not None:
            return
        model_name = config.memory.embedding_model
        try:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(model_name)
            _LOGGER.info("Loaded sentence-transformer model '%s' for embeddings", model_name)
        except Exception as exc:  # pragma: no cover - fallback path
            _LOGGER.warning("Falling back to OpenAI embeddings: %s", exc)
            from openai import OpenAI

            self._client = OpenAI(api_key=config.llm.api_key, base_url=config.llm.base_url)

    def embed(self, texts: Iterable[str]) -> np.ndarray:
        """Return embeddings for the provided texts."""

        self._ensure_model()
        texts_list: List[str] = [t.strip() for t in texts if t.strip()]
        if not texts_list:
            return np.empty((0, config.memory.embedding_dim))
        if self._model is not None:
            vectors = self._model.encode(texts_list, convert_to_numpy=True)
            return np.asarray(vectors, dtype=np.float32)
        if self._client is None:
            raise RuntimeError("Embedding service is not properly initialized")
        response = self._client.embeddings.create(model=config.memory.embedding_model, input=texts_list)
        vectors = [np.asarray(item.embedding, dtype=np.float32) for item in response.data]
        return np.vstack(vectors)


@functools.lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return a cached :class:`EmbeddingService` instance."""

    return EmbeddingService()
