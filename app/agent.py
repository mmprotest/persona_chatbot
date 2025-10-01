"""Core persona-driven conversational agent."""
from __future__ import annotations

import json
import logging
from typing import Dict, Iterator, List, Tuple

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
        self._scenario_prompt: str = ""
        self._scenario_turn_index: int | None = None
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
        self._scenario_turn_index = None
        if self._scenario_prompt:
            self._apply_scenario_prompt()
        _LOGGER.debug("Initialized conversation with system prompt")

    def _seed_persona_profile(self) -> None:
        if long_term.has_seed(self.persona_profile.seed_id):
            return
        for entry in self.persona_profile.seed_memories():
            long_term.add_memory(entry.role, entry.content, metadata=entry.metadata)

    def reset(self) -> None:
        self.conversation.clear()
        self._initialized = False
        self._scenario_turn_index = None

    def ingest_user_message(self, content: str) -> None:
        self._ensure_session()
        message = content.strip()
        if not message:
            _LOGGER.debug("Skipping empty user message ingestion")
            return
        turn = self.conversation.add("user", message)
        turn.id = long_term.add_memory("user", message, metadata={"type": "message"})

    def _gather_context_snippets(self, user_message: str) -> List[str]:
        memories = long_term.search_memories(user_message)
        snippets: List[str] = []
        for memory in memories:
            content = str(memory.get("content", "")).strip()
            if not content:
                continue
            role = str(memory.get("role", "memory")).strip() or "memory"
            snippets.append(f"{role}: {content}")
        return snippets

    def _build_runtime_guidance(self, context_snippets: List[str]) -> List[dict[str, str]]:
        instructions = (
            f"You are {self.persona.name}. Reply in the first person, stay grounded in the persona's "
            "voice, and keep continuity with the ongoing chat. Narrate any internal thinking in the "
            "first person as well, and let your answer feel considered and empathetic. It is "
            "mandatory to include a brief first-person inner monologue before every reply so the "
            "user can follow your live reasoning."
        )
        guidance: List[dict[str, str]] = [{"role": "system", "content": instructions}]
        if context_snippets:
            context_lines = "\n".join(f"- {snippet}" for snippet in context_snippets[:5])
            guidance.append(
                {
                    "role": "system",
                    "content": "Relevant memories you may draw from:\n" + context_lines,
                }
            )
        guidance.append(
            {
                "role": "system",
                "content": (
                    "Respond using XML tags exactly in this order:\n"
                    "<thinking>First-person inner monologue, 1-2 sentences. Always speak as 'I'."\
                    "</thinking>\n"
                    "<reply>Your spoken reply to the user, in the persona's voice.</reply>\n"
                    "<follow_up>First-person reminder that helps with the next turn.</follow_up>"
                ),
            }
        )
        return guidance

    def _format_context_summary(self, context_snippets: List[str]) -> str:
        if not context_snippets:
            return ""
        return "Relevant memories considered:\n" + "\n".join(f"- {snippet}" for snippet in context_snippets)

    def _parse_structured_reply(self, draft: str) -> Tuple[str, str, str]:
        candidate = draft.strip()

        def _extract(tag: str) -> str:
            open_tag = f"<{tag}>"
            close_tag = f"</{tag}>"
            start = candidate.find(open_tag)
            if start == -1:
                return ""
            start += len(open_tag)
            end = candidate.find(close_tag, start)
            if end == -1:
                return candidate[start:].strip()
            return candidate[start:end].strip()

        if "<reply>" in candidate:
            reflection = _extract("thinking")
            reply = _extract("reply")
            follow_up = _extract("follow_up")
            return reflection, reply, follow_up

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start != -1 and end != -1 and start < end:
            try:
                data = json.loads(candidate[start : end + 1])
                reflection = str(data.get("reflection", "")).strip()
                reply = str(data.get("reply", "")).strip()
                follow_up = str(data.get("follow_up", "")).strip()
                return reflection, reply, follow_up
            except (json.JSONDecodeError, TypeError, ValueError):
                _LOGGER.debug("Failed to parse structured reply", exc_info=True)
        return "", candidate, ""

    def _extract_tag_snapshot(self, payload: str, tag: str) -> Tuple[str | None, bool]:
        open_tag = f"<{tag}>"
        close_tag = f"</{tag}>"
        start = payload.find(open_tag)
        if start == -1:
            return None, False
        start += len(open_tag)
        end = payload.find(close_tag, start)
        if end == -1:
            return payload[start:], False
        return payload[start:end], True


    def _sanitize_reply(self, reply: str) -> str:
        """Remove accidental meta-commentary so the user only sees the response."""

        lines: List[str] = []
        blocked_prefixes = (
            "analysis:",
            "thoughts:",
            "inner thoughts:",
            "inner monologue:",
            "plan:",
            "reflection:",
            "reasoning:",
            "assistant's plan:",
        )
        for raw_line in reply.splitlines():
            line = raw_line.strip()
            if not line:
                lines.append("")
                continue
            lowered = line.lower()
            if lowered.startswith(blocked_prefixes):
                continue
            if lowered.startswith("[thinking") or lowered.startswith("(thinking"):
                continue
            lines.append(raw_line)
        sanitized = "\n".join(lines).strip()
        if sanitized:
            return sanitized
        return reply.strip()


    def set_scenario_prompt(self, scenario_prompt: str) -> None:
        """Update the active scenario context for the conversation."""

        normalized = scenario_prompt.strip()
        if normalized == self._scenario_prompt:
            return

        self._scenario_prompt = normalized
        if self._initialized:
            self._apply_scenario_prompt()
        else:
            self._scenario_turn_index = None

    def _apply_scenario_prompt(self) -> None:
        """Ensure the scenario context is represented in the conversation history."""

        scenario = self._scenario_prompt.strip()

        if not self._initialized:
            if not scenario:
                self._scenario_turn_index = None
            return

        if not scenario:
            if self._scenario_turn_index is not None and 0 <= self._scenario_turn_index < len(
                self.conversation.turns
            ):
                self.conversation.turns.pop(self._scenario_turn_index)
            self._scenario_turn_index = None
            return

        content = f"Scenario context: {scenario}"
        if self._scenario_turn_index is not None and 0 <= self._scenario_turn_index < len(
            self.conversation.turns
        ):
            self.conversation.update(self._scenario_turn_index, content)
            return

        insert_at = 1 if self.conversation.turns and self.conversation.turns[0].role == "system" else 0
        self.conversation.turns.insert(
            insert_at,
            ConversationTurn(role="system", content=content, editable=False),
        )
        self._scenario_turn_index = insert_at

    def generate_response(self, user_message: str, scenario_prompt: str | None = None) -> Dict[str, str]:
        user_message_clean = user_message.strip()
        if not user_message_clean:
            self._ensure_session()
            yield {
                "type": "complete",
                "result": {
                    "draft": "",
                    "final": "",
                    "reflection": "",
                    "context": "",
                    "plan": "",
                },
            }
            return

        if scenario_prompt is not None:
            self.set_scenario_prompt(scenario_prompt)

        if scenario_prompt is not None:
            self.set_scenario_prompt(scenario_prompt)

        self.ingest_user_message(user_message_clean)
        context_snippets = self._gather_context_snippets(user_message_clean)
        runtime_guidance = self._build_runtime_guidance(context_snippets)
        messages: List[dict[str, str]] = []
        total_turns = len(self.conversation.turns)
        for index, turn in enumerate(self.conversation.turns):
            if index == total_turns - 1 and runtime_guidance:
                messages.extend(runtime_guidance)
            messages.append({"role": turn.role, "content": turn.content})

        raw_response = self.llm.complete(messages, max_tokens=600)
        reflection, reply_body, follow_up = self._parse_structured_reply(raw_response)
        final_reply = self._sanitize_reply(reply_body.strip())
        if not final_reply:
            final_reply = self._sanitize_reply(raw_response.strip())

        assistant_turn = self.conversation.add("assistant", final_reply)
        assistant_turn.id = long_term.add_memory(
            "assistant", final_reply, metadata={"type": "message"}
        )

        context_summary = self._format_context_summary(context_snippets)
        plan = follow_up.strip()
        if plan:
            long_term.add_memory(
                "assistant_plan",
                plan,
                metadata={
                    "type": "forward_plan",
                    "seed_id": self.persona_profile.seed_id,
                },
            )

        return {
            "draft": final_reply,
            "final": final_reply,
            "reflection": reflection,
            "thinking": reflection.strip(),
            "context": context_summary,
            "plan": plan,
        }

    def generate_response(self, user_message: str) -> Dict[str, str]:
        final_result: Dict[str, str] | None = None
        for event in self.stream_response(user_message):
            if event.get("type") == "complete":
                result = event.get("result")
                if isinstance(result, dict):
                    final_result = result  # type: ignore[assignment]
        if final_result is None:
            return {
                "draft": "",
                "final": "",
                "reflection": "",
                "context": "",
                "plan": "",
            }
        return final_result

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
