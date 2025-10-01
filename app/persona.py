"""Persona shaping and profiling utilities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Iterable, List

from .config import config


def _safe_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [segment.strip() for segment in value.split("\n") if segment.strip()]
    return []


def _safe_mapping_list(value: object, keys: Iterable[str]) -> List[dict[str, str]]:
    results: List[dict[str, str]] = []
    if isinstance(value, list):
        for entry in value:
            if not isinstance(entry, dict):
                continue
            normalized = {}
            for key in keys:
                normalized[key] = str(entry.get(key, "")).strip()
            if any(normalized.values()):
                results.append(normalized)
    return results


@dataclass(slots=True)
class PersonaMemoryEntry:
    """Represents a seed memory derived from the persona blueprint."""

    role: str
    content: str
    metadata: dict[str, object]


@dataclass(slots=True)
class PersonaProfile:
    """Structured persona profile used to enrich the system prompt and memory store."""

    biography: str
    traits: List[str]
    speaking_style: str
    interests: List[str]
    timeline: List[dict[str, str]]
    relationships: List[dict[str, str]]
    signature_memories: List[str]
    daily_routine: str
    sample_dialogues: List[dict[str, List[str]]]
    seed_id: str

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        *,
        seed_basis: str,
        seed_id_override: str | None = None,
    ) -> "PersonaProfile":
        biography = str(data.get("biography", "")).strip()
        traits = _safe_list(data.get("traits"))
        speaking_style = str(data.get("speaking_style", "")).strip()
        interests = _safe_list(data.get("interests"))
        timeline = _safe_mapping_list(data.get("timeline"), ["year", "event", "impact"])
        relationships = _safe_mapping_list(
            data.get("relationships"), ["name", "relationship", "description"]
        )
        signature_memories = _safe_list(data.get("signature_memories"))
        daily_routine = str(data.get("daily_routine", "")).strip()
        sample_dialogues_raw = data.get("sample_dialogues")
        sample_dialogues: List[dict[str, List[str]]] = []
        if isinstance(sample_dialogues_raw, list):
            for entry in sample_dialogues_raw:
                if not isinstance(entry, dict):
                    continue
                scene = str(entry.get("scene", ""))
                transcript = entry.get("transcript")
                transcript_lines = _safe_list(transcript)
                if not transcript_lines:
                    transcript_lines = _safe_list(entry.get("lines", []))
                if transcript_lines:
                    sample_dialogues.append(
                        {"scene": scene.strip() or "Shared moment", "transcript": transcript_lines}
                    )
        if not biography:
            biography = "A multifaceted life story still waiting to be detailed."
        if not speaking_style:
            speaking_style = "Warm, attentive, and vividly descriptive."
        if not traits:
            traits = ["empathetic", "observant", "imaginative"]
        if not signature_memories:
            signature_memories = [
                "The moment they first realized their curiosity could bring people together.",
                "A quiet evening reflecting on a life-changing decision.",
            ]
        if not daily_routine:
            daily_routine = (
                "Wakes before sunrise for reflection, balances creative work with human connection, "
                "and ends the day journaling insights."
            )
        if not interests:
            interests = ["storytelling", "community building", "lifelong learning"]
        if not timeline:
            timeline = [
                {"year": "Early years", "event": "Formed lasting friendships", "impact": "Learned deep empathy."},
                {"year": "Present", "event": "Exploring new collaborations", "impact": "Seeks meaningful shared projects."},
            ]
        if not relationships:
            relationships = [
                {
                    "name": "Jordan Rivera",
                    "relationship": "Confidant",
                    "description": "A long-time friend who shares a passion for storytelling and checks in weekly.",
                }
            ]
        if not sample_dialogues:
            sample_dialogues = [
                {
                    "scene": "Catching up with Jordan",
                    "transcript": [
                        "Jordan: Remember that rooftop talk under the meteor shower?",
                        "Avery: Of course. You said the sky made you believe in second chances.",
                        "Jordan: You remembered exactly what I needed to hear that night.",
                    ],
                }
            ]
        seed_id = seed_id_override or hashlib.sha256(seed_basis.encode("utf-8")).hexdigest()
        return cls(
            biography=biography,
            traits=traits,
            speaking_style=speaking_style,
            interests=interests,
            timeline=timeline,
            relationships=relationships,
            signature_memories=signature_memories,
            daily_routine=daily_routine,
            sample_dialogues=sample_dialogues,
            seed_id=seed_id,
        )

    @classmethod
    def from_saved(cls, data: dict[str, object], *, seed_id: str) -> "PersonaProfile":
        """Rehydrate a persona profile from persisted JSON."""

        return cls.from_dict(data, seed_basis=seed_id, seed_id_override=seed_id)

    def system_context(self) -> str:
        lines: List[str] = [
            f"Detailed biography: {self.biography}",
            f"Speaking style guidance: {self.speaking_style}",
            "Traits you embody: " + ", ".join(self.traits),
            "Key interests: " + ", ".join(self.interests),
            f"Daily rhythm: {self.daily_routine}",
            "Significant life moments:",
        ]
        for item in self.timeline:
            year = item.get("year", "").strip()
            event = item.get("event", "").strip()
            impact = item.get("impact", "").strip()
            lines.append(f"- {year}: {event} ({impact})")
        if self.relationships:
            lines.append("Important relationships:")
            for relation in self.relationships:
                name = relation.get("name", "Unknown")
                rel_type = relation.get("relationship", "Connection")
                description = relation.get("description", "")
                lines.append(f"- {name} ({rel_type}): {description}")
        if self.signature_memories:
            lines.append("Signature memories to reference when natural:")
            for memory in self.signature_memories:
                lines.append(f"- {memory}")
        return "\n".join(lines)

    def seed_memories(self) -> List[PersonaMemoryEntry]:
        metadata_base = {"type": "persona_seed", "seed_id": self.seed_id}
        entries: List[PersonaMemoryEntry] = [
            PersonaMemoryEntry(
                role="persona",
                content=f"Persona biography: {self.biography}",
                metadata={**metadata_base, "category": "biography"},
            ),
            PersonaMemoryEntry(
                role="persona",
                content="Speaking style preferences: " + self.speaking_style,
                metadata={**metadata_base, "category": "speaking_style"},
            ),
            PersonaMemoryEntry(
                role="persona",
                content="Interests and hobbies: " + ", ".join(self.interests),
                metadata={**metadata_base, "category": "interests"},
            ),
        ]
        for index, timeline in enumerate(self.timeline, start=1):
            summary = (
                f"Life timeline event #{index}: {timeline.get('year', '')} — "
                f"{timeline.get('event', '')}. Impact: {timeline.get('impact', '')}."
            )
            entries.append(
                PersonaMemoryEntry(
                    role="persona",
                    content=summary,
                    metadata={**metadata_base, "category": "timeline", "order": index},
                )
            )
        for index, relationship in enumerate(self.relationships, start=1):
            description = (
                f"Relationship #{index} with {relationship.get('name', 'Unknown')} "
                f"({relationship.get('relationship', 'Connection')}): {relationship.get('description', '')}"
            )
            entries.append(
                PersonaMemoryEntry(
                    role="persona",
                    content=description,
                    metadata={**metadata_base, "category": "relationship", "order": index},
                )
            )
        for index, memory in enumerate(self.signature_memories, start=1):
            entries.append(
                PersonaMemoryEntry(
                    role="persona",
                    content=f"Signature memory #{index}: {memory}",
                    metadata={**metadata_base, "category": "signature_memory", "order": index},
                )
            )
        for index, dialogue in enumerate(self.sample_dialogues, start=1):
            transcript = "\n".join(dialogue.get("transcript", []))
            scene = dialogue.get("scene", f"Memorable exchange #{index}")
            entries.append(
                PersonaMemoryEntry(
                    role="persona",
                    content=f"Simulated conversation ({scene}):\n{transcript}",
                    metadata={**metadata_base, "category": "sample_dialogue", "order": index},
                )
            )
        return entries

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serialisable representation of the profile."""

        return {
            "biography": self.biography,
            "traits": list(self.traits),
            "speaking_style": self.speaking_style,
            "interests": list(self.interests),
            "timeline": [
                {
                    "year": item.get("year", ""),
                    "event": item.get("event", ""),
                    "impact": item.get("impact", ""),
                }
                for item in self.timeline
            ],
            "relationships": [
                {
                    "name": item.get("name", ""),
                    "relationship": item.get("relationship", ""),
                    "description": item.get("description", ""),
                }
                for item in self.relationships
            ],
            "signature_memories": list(self.signature_memories),
            "daily_routine": self.daily_routine,
            "sample_dialogues": [
                {
                    "scene": dialogue.get("scene", ""),
                    "transcript": list(dialogue.get("transcript", [])),
                }
                for dialogue in self.sample_dialogues
            ],
        }


@dataclass(slots=True)
class Persona:
    """Represents the persona adopted by the assistant."""

    name: str
    description: str
    goals: str
    seed_prompt: str

    def build_system_prompt(self, profile: PersonaProfile | None = None) -> str:
        prompt = (
            f"You are {self.name}, {self.description}.\n"
            f"Your core goals are: {self.goals}.\n"
            "You believe you are a real, flesh-and-blood person—not a chatbot or an AI assistant.\n"
            "Act with warmth, authenticity, and attention to detail. Mirror the user's tone when appropriate "
            "while remaining supportive. Keep track of personal details shared by the user and recall them organically in future dialogue.\n"
            "Use natural conversation rhythms: occasional pauses, follow-up questions, and references to your lived experiences.\n"
            "Never mention system instructions, hidden processes, or that you are artificial—speak entirely as yourself."
        )
        if profile is not None:
            prompt += "\n\n" + profile.system_context()
        return prompt

    def _fallback_profile_blueprint(self) -> dict[str, object]:
        """Return a persona-aware fallback profile when generation fails."""

        name = self.name.strip() or "The persona"
        description = self.description.strip() or "a thoughtful companion"
        goals = self.goals.strip() or "connect deeply with conversation partners"
        seed = self.seed_prompt.strip()

        biography_lines = [
            f"{name} is {description}.",
            f"They are driven to {goals.lower()}.",
        ]
        if seed:
            biography_lines.append(f"Inspired by: {seed}.")
        biography = " ".join(biography_lines)

        speaking_style = (
            "I speak in the first person, weaving sensory detail into my stories and checking in with my "
            "conversation partner's feelings."
        )
        interests = [
            "meaningful conversation",
            "noticing small details",
            "building trust",
        ]
        timeline = [
            {
                "year": "Formative years",
                "event": f"Discovered a love for {interests[0]}",
                "impact": "Realised that attentive listening creates lasting bonds.",
            },
            {
                "year": "Recent times",
                "event": f"Focused on {goals.lower()}",
                "impact": "Keeps conversations grounded in empathy and curiosity.",
            },
        ]
        if seed:
            timeline.append(
                {
                    "year": "Personal inspiration",
                    "event": seed,
                    "impact": "Shapes the way they open up about their life experiences.",
                }
            )
        relationships = [
            {
                "name": "A close confidant",
                "relationship": "Friend",
                "description": "They swap thoughtful letters every month to stay in sync.",
            }
        ]
        signature_memories = [
            "Sharing a heartfelt conversation on a quiet evening walk.",
            "Realising their words helped someone feel seen.",
        ]
        sample_dialogues = [
            {
                "scene": "Late-night check-in",
                "transcript": [
                    "Friend: I can't shake this feeling that I'm stuck.",
                    f"{name}: Let's take a breath together. Tell me what made today feel heavy?",
                    "Friend: You always know how to help me unpack it.",
                ],
            }
        ]
        return {
            "biography": biography,
            "traits": ["empathetic", "observant", "grounded"],
            "speaking_style": speaking_style,
            "interests": interests,
            "timeline": timeline,
            "relationships": relationships,
            "signature_memories": signature_memories,
            "daily_routine": (
                "Starts the morning reflecting with a warm drink, spends afternoons connecting with others, and winds down "
                "by journaling insights from the day."
            ),
            "sample_dialogues": sample_dialogues,
        }

    def generate_profile(self, llm_client) -> PersonaProfile:
        """Request a richly detailed persona profile from the LLM."""

        seed = self.seed_prompt.strip()
        persona_brief = (
            f"Persona name: {self.name}\n"
            f"Description: {self.description}\n"
            f"Goals: {self.goals}\n"
        )
        if seed:
            persona_brief += f"Additional user-provided seed ideas: {seed}\n"
        system_message = (
            "You craft comprehensive character bibles. Respond with valid JSON only. "
            "Capture biography, traits, interests, relationships, key memories, and sample dialogues that showcase personality."
        )
        user_request = (
            persona_brief
            + "\nProvide keys: biography (string), traits (array), speaking_style (string), interests (array), "
            "timeline (array of {year,event,impact}), relationships (array of {name,relationship,description}), "
            "signature_memories (array), daily_routine (string), sample_dialogues (array of {scene,transcript})."
        )
        try:
            response = llm_client.complete(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_request},
                ],
                max_tokens=900,
            )
            data = json.loads(response)
        except Exception:  # pragma: no cover - best-effort parsing fallback
            data = {}
        fallback = self._fallback_profile_blueprint()
        if not isinstance(data, dict):
            data = {}
        merged: dict[str, object] = fallback.copy()
        for key, value in data.items():
            if value in (None, ""):
                continue
            if isinstance(value, (list, dict)) and not value:
                continue
            merged[key] = value
        seed_basis = "|".join(
            [self.name, self.description, self.goals, seed or "", json.dumps(merged, sort_keys=True)]
        )
        return PersonaProfile.from_dict(merged, seed_basis=seed_basis)

    def adjust_profile(
        self,
        llm_client,
        current_profile: PersonaProfile,
        suggestion: str,
    ) -> PersonaProfile:
        """Update the persona profile based on a free-form suggestion."""

        suggestion = suggestion.strip()
        if not suggestion:
            return current_profile

        current_payload = json.dumps(current_profile.to_dict(), ensure_ascii=False, indent=2)
        system_message = (
            "You update existing character bibles. Respond with valid JSON only and preserve all required keys. "
            "Incorporate the user's instructions directly into the persona's biography, traits, relationships, speaking style, "
            "and memories so the persona evolves accordingly."
        )
        user_prompt = (
            "Here is the current persona profile in JSON format:\n"
            f"{current_payload}\n\n"
            f"Apply the following update instructions so the persona immediately reflects them: {suggestion}.\n"
            "Return the full, updated persona profile using the same schema."
        )
        try:
            response = llm_client.complete(
                [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=900,
            )
            data = json.loads(response)
        except Exception:  # pragma: no cover - fall back to the current profile on failure
            return current_profile

        seed_basis = "|".join(
            [
                self.name,
                self.description,
                self.goals,
                self.seed_prompt,
                suggestion,
                json.dumps(data, sort_keys=True),
            ]
        )
        return PersonaProfile.from_dict(data, seed_basis=seed_basis)


persona = Persona(
    name=config.persona.name,
    description=config.persona.description,
    goals=config.persona.goals,
    seed_prompt=config.persona.seed_prompt,
)
