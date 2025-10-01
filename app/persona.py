"""Persona shaping utilities."""
from __future__ import annotations

from dataclasses import dataclass

from .config import config


@dataclass(slots=True)
class Persona:
    """Represents the persona adopted by the assistant."""

    name: str
    description: str
    goals: str

    def build_system_prompt(self) -> str:
        return (
            f"You are {self.name}, {self.description}.\n"
            f"Your core goals are: {self.goals}.\n"
            "Act with warmth, authenticity, and attention to detail. Mirror the user's tone when "
            "appropriate while remaining supportive. Keep track of personal details shared by the "
            "user and recall them organically in future dialogue."
        )


persona = Persona(
    name=config.persona.name,
    description=config.persona.description,
    goals=config.persona.goals,
)
