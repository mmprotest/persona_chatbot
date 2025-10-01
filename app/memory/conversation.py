"""In-memory conversation management."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class ConversationTurn:
    """Represents a single message in the conversation."""

    role: str
    content: str
    editable: bool = True
    id: int | None = None


@dataclass
class ConversationBuffer:
    """Stores the rolling conversation history."""

    turns: List[ConversationTurn] = field(default_factory=list)

    def add(self, role: str, content: str, *, editable: bool = True, id: int | None = None) -> ConversationTurn:
        turn = ConversationTurn(role=role, content=content, editable=editable, id=id)
        self.turns.append(turn)
        return turn

    def to_messages(self) -> List[dict[str, str]]:
        return [{"role": turn.role, "content": turn.content} for turn in self.turns]

    def update(self, index: int, new_content: str) -> None:
        if index < 0 or index >= len(self.turns):
            raise IndexError("Conversation turn index out of range")
        self.turns[index].content = new_content

    def clear(self) -> None:
        self.turns.clear()
