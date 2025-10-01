"""Persona agent and persistence utilities.

This module contains a lightweight `PersonaAgent` that stores persona
memories in a SQLite database.  The real project this exercise is inspired
by contains a considerably larger surface area, but the behaviour required
for the regression fixed in this kata is captured here so that it can be
unit tested without the original infrastructure.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import sqlite3
import uuid
from typing import Iterable, Optional, Sequence


SEED_CATEGORIES = ("biography", "timeline", "relationship", "persona_profile")


@dataclass
class PersonaSuggestion:
    """Container holding the data required to reseed a persona profile."""

    persona_id: str
    biography: str
    timeline: Sequence[str]
    relationships: Sequence[str]
    profile_summary: str
    seed_id: Optional[str] = None
    extra_memories: Sequence[str] = field(default_factory=list)

    def ensure_seed_id(self) -> str:
        if self.seed_id is None:
            self.seed_id = str(uuid.uuid4())
        return self.seed_id


class MemoryStore:
    """Simple SQLite wrapper used by :class:`PersonaAgent`."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                persona_id TEXT NOT NULL,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                seed_id TEXT
            )
            """
        )
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_memories_persona_seed
            ON memories(persona_id, seed_id, category)
            """
        )
        self.connection.commit()

    def has_seed(
        self,
        persona_id: str,
        seed_id: str,
        *,
        skip_categories: Iterable[str] = (),
    ) -> bool:
        """Return ``True`` when any memories exist for ``persona_id`` and ``seed_id``.

        Parameters
        ----------
        persona_id:
            The persona identifier.
        seed_id:
            The seed identifier that should be checked.
        skip_categories:
            Optional collection of categories that should be ignored when
            determining whether a seed exists.  This supports the regression
            fix where we need to ignore the ``persona_profile`` summary when
            deciding if the biography/timeline/relationship rows were
            generated.
        """

        skip = tuple(skip_categories)
        placeholders = ""
        params: list[object] = [persona_id, seed_id]
        if skip:
            placeholders = " AND category NOT IN (%s)" % ", ".join("?" for _ in skip)
            params.extend(skip)

        cursor = self.connection.cursor()
        cursor.execute(
            f"SELECT 1 FROM memories WHERE persona_id = ? AND seed_id = ?{placeholders} LIMIT 1",
            params,
        )
        return cursor.fetchone() is not None

    def replace_seed_memories(
        self,
        persona_id: str,
        seed_id: str,
        category: str,
        entries: Iterable[str],
    ) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "DELETE FROM memories WHERE persona_id = ? AND category = ?",
            (persona_id, category),
        )
        cursor.executemany(
            "INSERT INTO memories (persona_id, category, content, seed_id) VALUES (?, ?, ?, ?)",
            ((persona_id, category, entry, seed_id) for entry in entries),
        )
        self.connection.commit()

    def write_memories(
        self,
        persona_id: str,
        memories: Iterable[str],
        *,
        category: str = "memory",
        seed_id: Optional[str] = None,
    ) -> None:
        cursor = self.connection.cursor()
        cursor.executemany(
            "INSERT INTO memories (persona_id, category, content, seed_id) VALUES (?, ?, ?, ?)",
            ((persona_id, category, memory, seed_id) for memory in memories),
        )
        self.connection.commit()


class PersonaAgent:
    """Agent responsible for seeding persona memories."""

    def __init__(self, memory_store: MemoryStore) -> None:
        self.memory_store = memory_store

    def _seed_persona_profile(self, suggestion: PersonaSuggestion) -> None:
        seed_id = suggestion.ensure_seed_id()
        self.memory_store.replace_seed_memories(
            suggestion.persona_id,
            seed_id,
            "biography",
            [suggestion.biography],
        )
        self.memory_store.replace_seed_memories(
            suggestion.persona_id,
            seed_id,
            "timeline",
            suggestion.timeline,
        )
        self.memory_store.replace_seed_memories(
            suggestion.persona_id,
            seed_id,
            "relationship",
            suggestion.relationships,
        )
        self.memory_store.replace_seed_memories(
            suggestion.persona_id,
            seed_id,
            "persona_profile",
            [suggestion.profile_summary],
        )

    def apply_persona_suggestion(self, suggestion: PersonaSuggestion) -> None:
        """Persist persona information from ``suggestion``.

        The regression we are protecting against here occurred when the
        ``persona_profile`` summary was written first.  The subsequent check
        for whether the seed existed saw the summary and incorrectly assumed
        the biography/timeline/relationship rows had already been written,
        leaving those categories stale.  We now seed the profile *before*
        persisting any non-seed memories and explicitly ignore the summary
        when determining if the seed is present.
        """

        seed_id = suggestion.ensure_seed_id()

        # Ensure the structured persona profile rows are rewritten before any
        # additional persona memories are appended.
        if not self.memory_store.has_seed(
            suggestion.persona_id,
            seed_id,
            skip_categories=("persona_profile",),
        ):
            self._seed_persona_profile(suggestion)
        else:
            # Even if the main seed categories exist we still refresh the
            # profile summary to keep it in sync with the suggestion.
            self.memory_store.replace_seed_memories(
                suggestion.persona_id,
                seed_id,
                "persona_profile",
                [suggestion.profile_summary],
            )

        if suggestion.extra_memories:
            self.memory_store.write_memories(
                suggestion.persona_id,
                suggestion.extra_memories,
                seed_id=seed_id,
            )
