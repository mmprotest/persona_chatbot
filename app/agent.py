"""Core persona-driven conversational agent."""
from __future__ import annotations

import logging
from typing import Dict, List

from .llm.factory import create_llm_client
from .memory import long_term
from .memory.conversation import ConversationBuffer, ConversationTurn
from .persona import Persona, PersonaProfile, persona
from .persona_store import upsert_persona

_LOGGER = logging.getLogger(__name__)


class PersonaAgent:
    """Agent that blends persona adherence with adaptive reasoning."""

    def __init__(
        self,
        persona_config: Persona | None = None,
        persona_profile: PersonaProfile | None = None,
    ) -> None:
        self.llm = create_llm_client()
        self.conversation = ConversationBuffer()
        self._initialized = False
        self.persona = persona_config or persona
        if persona_profile is None:
            self.persona_profile = self.persona.generate_profile(self.llm)
        else:
            self.persona_profile = persona_profile
        self.persona_record_id = upsert_persona(self.persona, self.persona_profile)
        self._seed_persona_profile()

    def _ensure_session(self) -> None:
        if self._initialized:
            return
        system_prompt = self.persona.build_system_prompt(self.persona_profile)
        self.conversation.add("system", system_prompt, editable=False)
        self._initialized = True
        _LOGGER.debug("Initialized conversation with system prompt")

    def _seed_persona_profile(self) -> None:
        if long_term.has_seed(self.persona_profile.seed_id):
            return
        for entry in self.persona_profile.seed_memories():
            long_term.add_memory(entry.role, entry.content, metadata=entry.metadata)

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
        plan = self._forecast_next_steps(user_message, improved)
        if plan.strip():
            long_term.add_memory(
                "assistant_plan",
                plan,
                metadata={
                    "type": "forward_plan",
                    "seed_id": self.persona_profile.seed_id,
                },
            )
        return {
            "draft": draft,
            "final": improved,
            "reflection": reflection,
            "context": context_prompt,
            "plan": plan,
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

    def _forecast_next_steps(self, user_message: str, assistant_reply: str) -> str:
        prompt = (
            "Consider the persona's biography, interests, and relationships. The user just said: "
            f"{user_message!r}. The assistant replied: {assistant_reply!r}. "
            "Outline 2-3 actionable bullet points describing how the assistant should nurture the relationship, "
            "anticipate future topics, or suggest follow-up questions. Be specific and stay in character."
        )
        return self.llm.reflect(prompt, max_tokens=300)

    def apply_persona_suggestion(self, suggestion: str) -> PersonaProfile:
        """Apply a user-provided suggestion to evolve the persona in real time."""

        suggestion = suggestion.strip()
        if not suggestion:
            return self.persona_profile

        updated_profile = self.persona.adjust_profile(self.llm, self.persona_profile, suggestion)
        if not isinstance(updated_profile, PersonaProfile):
            return self.persona_profile

        self.persona_profile = updated_profile
        self.persona_record_id = upsert_persona(self.persona, self.persona_profile)

        if self._initialized and self.conversation.turns:
            new_prompt = self.persona.build_system_prompt(self.persona_profile)
            first_turn = self.conversation.turns[0]
            if first_turn.role == "system":
                self.conversation.update(0, new_prompt)
            else:
                self.conversation.turns.insert(
                    0, ConversationTurn(role="system", content=new_prompt, editable=False)
                )
        else:
            self._initialized = False
            self.conversation.clear()

        long_term.add_memory(
            "persona_update",
            suggestion,
            metadata={
                "type": "persona_update_instruction",
                "seed_id": self.persona_profile.seed_id,
            },
        )
        long_term.add_memory(
            "persona",
            "Updated persona profile summary:\n" + self.persona_profile.system_context(),
            metadata={
                "type": "persona_profile",
                "seed_id": self.persona_profile.seed_id,
            },
        )
        self._seed_persona_profile()
        return self.persona_profile


def create_agent(
    persona_config: Persona | None = None,
    persona_profile: PersonaProfile | None = None,
) -> PersonaAgent:
    return PersonaAgent(persona_config=persona_config, persona_profile=persona_profile)
