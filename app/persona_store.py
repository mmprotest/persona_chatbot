"""Persistence utilities for persona profiles."""
from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List, Optional

from .config import config
from .persona import Persona, PersonaProfile

_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS personas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    goals TEXT NOT NULL,
    seed_prompt TEXT,
    profile_json TEXT NOT NULL,
    seed_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_personas_name ON personas(name);
"""


def _ensure_database(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(path)) as conn:
        conn.executescript(_TABLE_SCHEMA)


@contextmanager
def _connect() -> Iterable[sqlite3.Connection]:
    db_path = Path(config.memory.database_path)
    _ensure_database(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        yield conn
    finally:
        conn.close()


def upsert_persona(persona: Persona, profile: PersonaProfile) -> int:
    """Insert or update a persona record and return its identifier."""

    payload = json.dumps(profile.to_dict(), ensure_ascii=False)
    now = time.time()
    with _connect() as conn:
        row = conn.execute("SELECT id FROM personas WHERE name = ?", (persona.name,)).fetchone()
        if row:
            persona_id = int(row[0])
            conn.execute(
                "UPDATE personas SET description = ?, goals = ?, seed_prompt = ?, profile_json = ?, seed_id = ?, updated_at = ? WHERE id = ?",
                (
                    persona.description,
                    persona.goals,
                    persona.seed_prompt,
                    payload,
                    profile.seed_id,
                    now,
                    persona_id,
                ),
            )
            conn.commit()
            return persona_id
        cursor = conn.execute(
            "INSERT INTO personas (name, description, goals, seed_prompt, profile_json, seed_id, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                persona.name,
                persona.description,
                persona.goals,
                persona.seed_prompt,
                payload,
                profile.seed_id,
                now,
                now,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def list_personas() -> List[dict[str, object]]:
    """Return all stored persona records sorted by recency."""

    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, name, description, goals, seed_prompt, profile_json, seed_id, created_at, updated_at FROM personas ORDER BY updated_at DESC"
        ).fetchall()
    personas: List[dict[str, object]] = []
    for row in rows:
        profile_data = json.loads(row[5]) if row[5] else {}
        personas.append(
            {
                "id": int(row[0]),
                "name": str(row[1]),
                "description": str(row[2]),
                "goals": str(row[3]),
                "seed_prompt": str(row[4]) if row[4] else "",
                "profile": profile_data,
                "seed_id": str(row[6]),
                "created_at": float(row[7]),
                "updated_at": float(row[8]),
            }
        )
    return personas


def load_persona(persona_id: int) -> Optional[dict[str, object]]:
    """Load a persona record by identifier."""

    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, description, goals, seed_prompt, profile_json, seed_id, created_at, updated_at FROM personas WHERE id = ?",
            (persona_id,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": int(row[0]),
        "name": str(row[1]),
        "description": str(row[2]),
        "goals": str(row[3]),
        "seed_prompt": str(row[4]) if row[4] else "",
        "profile": json.loads(row[5]) if row[5] else {},
        "seed_id": str(row[6]),
        "created_at": float(row[7]),
        "updated_at": float(row[8]),
    }


def find_persona_by_name(name: str) -> Optional[dict[str, object]]:
    """Return the record for the specified persona name if it exists."""

    with _connect() as conn:
        row = conn.execute(
            "SELECT id, name, description, goals, seed_prompt, profile_json, seed_id, created_at, updated_at FROM personas WHERE name = ?",
            (name,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": int(row[0]),
        "name": str(row[1]),
        "description": str(row[2]),
        "goals": str(row[3]),
        "seed_prompt": str(row[4]) if row[4] else "",
        "profile": json.loads(row[5]) if row[5] else {},
        "seed_id": str(row[6]),
        "created_at": float(row[7]),
        "updated_at": float(row[8]),
    }

