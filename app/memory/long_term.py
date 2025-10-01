"""Persistent long-term memory backed by SQLite with vector search."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

import numpy as np

from ..config import config
from .embeddings import get_embedding_service

_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at REAL NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata TEXT,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_memories_created_at ON memories(created_at DESC);
"""


def _ensure_database(path: str) -> None:
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(_DB_SCHEMA)


@contextmanager
def _connect() -> Iterable[sqlite3.Connection]:
    db_path = config.memory.database_path
    _ensure_database(db_path)
    conn = sqlite3.connect(db_path)
    try:
        yield conn
    finally:
        conn.close()


def _array_to_blob(array: np.ndarray) -> bytes:
    return array.astype(np.float32).tobytes()


def _blob_to_array(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def add_memory(role: str, content: str, metadata: dict | None = None) -> int:
    """Persist a memory entry with its embedding."""

    content = content.strip()
    if not content:
        raise ValueError("Cannot store empty memory content")
    embedding_service = get_embedding_service()
    vector = embedding_service.embed([content])
    if vector.size == 0:
        raise RuntimeError("Failed to compute embedding for memory content")
    payload = json.dumps(metadata or {})
    with _connect() as conn:
        cursor = conn.execute(
            "INSERT INTO memories (created_at, role, content, metadata, embedding) VALUES (?, ?, ?, ?, ?)",
            (time.time(), role, content, payload, _array_to_blob(vector[0])),
        )
        conn.commit()
        return int(cursor.lastrowid)


def update_memory(memory_id: int, role: str, content: str, metadata: dict | None = None) -> None:
    """Update an existing memory entry and refresh its embedding."""

    content = content.strip()
    if not content:
        raise ValueError("Cannot store empty memory content")
    embedding_service = get_embedding_service()
    vector = embedding_service.embed([content])
    if vector.size == 0:
        raise RuntimeError("Failed to compute embedding for memory content")
    payload = json.dumps(metadata or {})
    with _connect() as conn:
        conn.execute(
            "UPDATE memories SET role = ?, content = ?, metadata = ?, embedding = ? WHERE id = ?",
            (role, content, payload, _array_to_blob(vector[0]), memory_id),
        )
        conn.commit()


def fetch_recent(limit: int = 20) -> List[tuple[int, float, str, str, dict]]:
    """Return the most recent memories."""

    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, created_at, role, content, metadata FROM memories ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    results: List[tuple[int, float, str, str, dict]] = []
    for row in rows:
        metadata = json.loads(row[4]) if row[4] else {}
        results.append((int(row[0]), float(row[1]), str(row[2]), str(row[3]), metadata))
    return results


def search_memories(query: str, limit: int | None = None) -> List[dict]:
    """Return memories ordered by cosine similarity to the query."""

    limit = limit or config.memory.max_results
    embedding_service = get_embedding_service()
    query_vec = embedding_service.embed([query])
    if query_vec.size == 0:
        return []
    q = query_vec[0]
    with _connect() as conn:
        rows = conn.execute("SELECT id, role, content, metadata, embedding FROM memories").fetchall()
    memories: List[dict] = []
    for row in rows:
        emb = _blob_to_array(row[4])
        if emb.size == 0:
            continue
        denom = np.linalg.norm(q) * np.linalg.norm(emb)
        similarity = float(np.dot(q, emb) / denom) if denom else 0.0
        if similarity < config.memory.relevance_threshold:
            continue
        metadata = json.loads(row[3]) if row[3] else {}
        memories.append(
            {
                "id": int(row[0]),
                "role": str(row[1]),
                "content": str(row[2]),
                "metadata": metadata,
                "similarity": similarity,
            }
        )
    memories.sort(key=lambda item: item["similarity"], reverse=True)
    return memories[:limit]
