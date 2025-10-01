"""Core persona-driven conversational agent."""
from __future__ import annotations

import logging
from typing import Dict, List

from .llm.factory import create_llm_client
from .memory import long_term
from .memory.conversation import ConversationBuffer
from .persona import persona

_LOGGER = logging.getLogger(__name__)


class PersonaAgent:
    """Agent that blends persona adherence with adaptive reasoning."""

    def __init__(self) -> None:
        self.llm = create_llm_client()
        self.conversation = ConversationBuffer()
        self._initialized = False

    def _ensure_session(self) -> None:
        if self._initialized:
            return
        system_prompt = persona.build_system_prompt()
        self.conversation.add("system", system_prompt, editable=False)
        self._initialized = True
        _LOGGER.debug("Initialized conversation with system prompt")

    def reset(self) -> None:
        self.conversation.clear()
        self._initialized = False

    def ingest_user_message(self, content: str) -> None:
        self._ensure_session()
        turn = self.conversation.add("user", content)
        turn.id = long_term.add_memory("user", content, metadata={"type": "message"})

    def _build_contextual_prompt(self, user_message: str) -> str:
        memories = long_term.search_memories(user_message)
        if not memories:
            return ""
        context_lines = [
            "The user previously shared the following relevant information:"
        ]
        for memory in memories:
            context_lines.append(f"- ({memory['similarity']:.2f}) {memory['role']}: {memory['content']}")
        return "\n".join(context_lines)

    def _generate_reflection(self, user_message: str, assistant_draft: str) -> str:
        prompt = (
            "You are evaluating an assistant's draft reply. Consider the persona, the latest user "
            f"message: {user_message!r}, and the assistant draft: {assistant_draft!r}. "
            "Provide bullet-point guidance on how to improve tone, incorporate past memories, and "
            "anticipate follow-up questions."
        )
        reflection = self.llm.reflect(prompt, max_tokens=256)
        return reflection

    def _apply_reflection(self, reflection: str, assistant_draft: str) -> str:
        messages = [
            {
                "role": "system",
                "content": (
                    "You are refining an assistant response based on reflection notes. Ensure the "
                    "voice matches the persona, and weave in relevant memories when natural."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Reflection notes:\n{reflection}\n\n"
                    f"Original draft response:\n{assistant_draft}\n\n"
                    "Produce an improved final reply."
                ),
            },
        ]
        improved = self.llm.complete(messages, max_tokens=512)
        return improved

    def generate_response(self, user_message: str) -> Dict[str, str]:
        self.ingest_user_message(user_message)
        user_message = user_message.strip()
        context_prompt = self._build_contextual_prompt(user_message)
        messages: List[dict[str, str]] = list(self.conversation.to_messages())
        if context_prompt:
            messages.append({"role": "system", "content": context_prompt})
        messages.append(
            {
                "role": "system",
                "content": (
                    "Before responding, take a breath, think through the user's intentions, and aim "
                    "for an immersive, emotionally intelligent reply."
                ),
            }
        )
        draft = self.llm.complete(messages, max_tokens=800)
        reflection = self._generate_reflection(user_message, draft)
        improved = self._apply_reflection(reflection, draft)
        assistant_turn = self.conversation.add("assistant", improved)
        assistant_turn.id = long_term.add_memory("assistant", improved, metadata={"type": "message"})
        long_term.add_memory(
            "assistant_reflection",
            reflection,
            metadata={"type": "reflection", "source": "self"},
        )
        return {
            "draft": draft,
            "final": improved,
            "reflection": reflection,
            "context": context_prompt,
        }

    def edit_turn(self, index: int, new_content: str) -> None:
        self.conversation.update(index, new_content)
        turn = self.conversation.turns[index]
        if turn.id is not None:
            long_term.update_memory(turn.id, turn.role, new_content, metadata={"type": "message", "edited": True})

    def load_recent_memories(self) -> List[dict]:
        records = long_term.fetch_recent(limit=20)
        return [
            {
                "id": record[0],
                "created_at": record[1],
                "role": record[2],
                "content": record[3],
                "metadata": record[4],
            }
            for record in records
        ]


def create_agent() -> PersonaAgent:
    return PersonaAgent()
