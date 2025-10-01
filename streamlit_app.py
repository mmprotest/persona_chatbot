"""Streamlit UI for the persona-driven chatbot."""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from app.agent import create_agent
from app.persona import Persona, PersonaProfile, persona as default_persona
from app.persona_store import (
    delete_persona,
    find_persona_by_name,
    list_personas,
    load_persona,
    upsert_persona,
)


def _rerun() -> None:
    """Trigger a Streamlit rerun compatible with newer and older versions."""
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - support for older Streamlit releases
        st.experimental_rerun()


st.set_page_config(page_title="Persona Chatbot", page_icon="🧠", layout="wide")


def _hydrate_persona(record: dict[str, object]) -> tuple[Persona, PersonaProfile]:
    persona_config = Persona(
        name=str(record.get("name", "")),
        description=str(record.get("description", "")),
        goals=str(record.get("goals", "")),
        seed_prompt=str(record.get("seed_prompt", "")),
    )
    profile_payload = record.get("profile") if isinstance(record.get("profile"), dict) else {}
    profile = PersonaProfile.from_saved(profile_payload or {}, seed_id=str(record.get("seed_id", "")))
    return persona_config, profile


def _split_nonempty_lines(raw: str) -> list[str]:
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _parse_timeline_input(raw: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        year, event, impact = (parts + ["", "", ""])[:3]
        if any([year, event, impact]):
            entries.append({"year": year, "event": event, "impact": impact})
    return entries


def _parse_relationships_input(raw: str) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split("|")]
        name, relationship, description = (parts + ["", "", ""])[:3]
        if any([name, relationship, description]):
            entries.append(
                {"name": name, "relationship": relationship, "description": description}
            )
    return entries


def _parse_sample_dialogues_input(raw: str, fallback_name: str) -> list[dict[str, list[str]]]:
    entries: list[dict[str, list[str]]] = []
    blocks = [block.strip() for block in raw.split("\n\n") if block.strip()]
    for block in blocks:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue
        first = lines[0]
        scene = "Shared moment"
        transcript_lines = lines
        if first.lower().startswith("scene:"):
            scene = first.split(":", 1)[1].strip() or "Shared moment"
            transcript_lines = [line for line in lines[1:] if line]
        if not transcript_lines:
            continue
        entries.append({"scene": scene, "transcript": transcript_lines})
    if not entries:
        entries.append(
            {
                "scene": "Quick check-in",
                "transcript": [
                    f"Friend: Hey {fallback_name}, how are you holding up today?",
                    f"{fallback_name}: I'm steady. What's been on your mind?",
                ],
            }
        )
    return entries


def _profile_from_manual_inputs(
    *,
    persona_name: str,
    description: str,
    goals: str,
    biography: str,
    speaking_style: str,
    traits_text: str,
    interests_text: str,
    daily_routine: str,
    timeline_text: str,
    relationships_text: str,
    memories_text: str,
    dialogues_text: str,
    seed_id_override: str | None = None,
) -> PersonaProfile:
    profile_payload = {
        "biography": biography.strip(),
        "traits": _split_nonempty_lines(traits_text),
        "speaking_style": speaking_style.strip(),
        "interests": _split_nonempty_lines(interests_text),
        "daily_routine": daily_routine.strip(),
        "timeline": _parse_timeline_input(timeline_text),
        "relationships": _parse_relationships_input(relationships_text),
        "signature_memories": _split_nonempty_lines(memories_text),
        "sample_dialogues": _parse_sample_dialogues_input(dialogues_text, persona_name),
    }
    seed_basis = "|".join(
        [
            persona_name,
            description,
            goals,
            biography,
            speaking_style,
            "manual",
            str(time.time()),
        ]
    )
    kwargs = {"seed_basis": seed_basis}
    if seed_id_override:
        kwargs["seed_id_override"] = seed_id_override
    return PersonaProfile.from_dict(profile_payload, **kwargs)


def _timeline_to_text(timeline: list[dict[str, str]]) -> str:
    lines = []
    for entry in timeline:
        year = str(entry.get("year", "")).strip()
        event = str(entry.get("event", "")).strip()
        impact = str(entry.get("impact", "")).strip()
        lines.append(" | ".join(part for part in [year, event, impact] if part))
    return "\n".join(line for line in lines if line)


def _relationships_to_text(relationships: list[dict[str, str]]) -> str:
    lines = []
    for entry in relationships:
        name = str(entry.get("name", "")).strip()
        relationship = str(entry.get("relationship", "")).strip()
        description = str(entry.get("description", "")).strip()
        lines.append(" | ".join(part for part in [name, relationship, description] if part))
    return "\n".join(line for line in lines if line)


def _dialogues_to_text(dialogues: list[dict[str, list[str]]]) -> str:
    blocks: list[str] = []
    for entry in dialogues:
        scene = str(entry.get("scene", "Shared moment")).strip()
        transcript = entry.get("transcript") or []
        lines = [f"Scene: {scene}"]
        lines.extend(line for line in transcript if line)
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _set_agent(agent) -> None:
    st.session_state.agent = agent
    st.session_state.active_persona_id = getattr(agent, "persona_record_id", None)
    if "editing" not in st.session_state:
        st.session_state.editing = {}
    else:
        st.session_state.editing.clear()
    st.session_state.last_generation = None


def set_active_persona(persona_id: int) -> None:
    record = load_persona(persona_id)
    if not record:
        st.warning("Unable to load the selected persona.")
        return
    persona_config, profile = _hydrate_persona(record)
    agent = create_agent(persona_config=persona_config, persona_profile=profile)
    _set_agent(agent)
    _rerun()


def get_agent():
    if "agent" not in st.session_state:
        active_id = st.session_state.get("active_persona_id")
        record: Optional[dict[str, object]] = None
        if active_id is not None:
            record = load_persona(int(active_id))
        if record is None:
            record = find_persona_by_name(default_persona.name)
        if record:
            persona_config, profile = _hydrate_persona(record)
            agent = create_agent(persona_config=persona_config, persona_profile=profile)
        else:
            agent = create_agent()
        _set_agent(agent)
    if "editing" not in st.session_state:
        st.session_state.editing = {}
    if "last_generation" not in st.session_state:
        st.session_state.last_generation = None
    return st.session_state.agent


def toggle_edit(index: int, value: bool) -> None:
    st.session_state.editing[index] = value


def _format_timestamp(ts: float) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
    except Exception:  # pragma: no cover - fallback for malformed timestamps
        return "Unknown"


def handle_edit(index: int, turn_content: str) -> None:
    agent = get_agent()
    agent.edit_turn(index, turn_content)
    toggle_edit(index, False)
    _rerun()


def render_sidebar() -> None:
    agent = get_agent()
    with st.sidebar:
        st.header("Session Controls")
        if st.button("Reset Conversation", use_container_width=True):
            agent.reset()
            st.session_state.last_generation = None
            if "editing" in st.session_state:
                st.session_state.editing.clear()
            _rerun()
        st.divider()
        st.subheader("Persona Library")
        personas = list_personas()
        persona_options = [record["id"] for record in personas]
        current_id = st.session_state.get("active_persona_id")
        try:
            current_id = int(current_id) if current_id is not None else None
        except (TypeError, ValueError):  # pragma: no cover - defensive parsing
            current_id = None
        if persona_options:
            try:
                index = persona_options.index(current_id) if current_id in persona_options else 0
            except ValueError:
                index = 0
            selected_id = st.selectbox(
                "Active persona",
                persona_options,
                index=index,
                format_func=lambda pid: next(
                    (f"{item['name']} — {item['description']}" for item in personas if item["id"] == pid),
                    str(pid),
                ),
                key="persona_selector",
            )
            if selected_id != current_id:
                set_active_persona(int(selected_id))
        else:
            st.info("No personas saved yet. Visit the Persona Studio tab to craft one.")
        st.caption("Manage personas in the Persona Studio tab of the main view.")
        st.divider()
        st.subheader("Persona Profile")
        profile = agent.persona_profile
        st.markdown(
            f"**{profile.biography}**\n\n"
            f"**Traits:** {', '.join(profile.traits)}\n\n"
            f"**Speaking style:** {profile.speaking_style}\n\n"
            f"**Interests:** {', '.join(profile.interests)}\n\n"
            f"**Daily rhythm:** {profile.daily_routine}",
        )
        if st.session_state.get("clear_persona_suggestion_input"):
            st.session_state.pop("clear_persona_suggestion_input", None)
            st.session_state.persona_suggestion_input = ""

        with st.form("persona_suggestion_form", clear_on_submit=False):
            suggestion = st.text_area(
                "Suggest persona adjustments",
                key="persona_suggestion_input",
                help=(
                    "Describe how you'd like the persona to evolve. The agent will rewrite its backstory, "
                    "traits, and seed memories to match."
                ),
                height=120,
            )
            submitted = st.form_submit_button("Apply Persona Update", use_container_width=True)
        if submitted:
            if suggestion.strip():
                agent.apply_persona_suggestion(suggestion)
                st.session_state.clear_persona_suggestion_input = True
                st.session_state.last_generation = None
                _rerun()
            else:
                st.warning("Enter a suggestion before applying the update.")
        st.divider()
        st.subheader("Recent Memories")
        for record in agent.load_recent_memories():
            timestamp = _format_timestamp(record["created_at"])
            st.markdown(
                f"**{record['role']}** · {timestamp}\n\n"
                f"{record['content']}\n\n"
                f"<small>{record['metadata']}</small>",
                unsafe_allow_html=True,
            )
        if st.session_state.last_generation:
            st.divider()
            st.subheader("Self-Reflection")
            st.markdown(st.session_state.last_generation.get("reflection", "_No reflection yet._"))
            st.divider()
            st.subheader("Context Used")
            st.markdown(st.session_state.last_generation.get("context", "_No additional context._"))
            st.divider()
            st.subheader("Forward Plan")
            st.markdown(st.session_state.last_generation.get("plan", "_No plan computed._"))


def render_conversation(agent) -> None:
    """Render the conversation with optional editing controls."""

    for index, turn in enumerate(agent.conversation.turns):
        with st.chat_message(turn.role):
            is_editing = st.session_state.editing.get(index, False)
            if is_editing and turn.editable:
                new_content = st.text_area(
                    "Edit message",
                    turn.content,
                    key=f"edit_area_{index}",
                    height=150,
                )
                col_save, col_cancel = st.columns(2)
                with col_save:
                    if st.button("Save", key=f"save_{index}"):
                        handle_edit(index, new_content)
                        return
                with col_cancel:
                    if st.button("Cancel", key=f"cancel_{index}"):
                        toggle_edit(index, False)
                        _rerun()
                        return
            else:
                st.markdown(turn.content)
                if turn.editable:
                    if st.button("Edit", key=f"edit_{index}"):
                        toggle_edit(index, True)
                        _rerun()
                        return


def render_persona_studio(agent) -> None:
    """Render creation, editing, and deletion tools for personas."""

    if message := st.session_state.pop("persona_studio_message", None):
        st.success(message)
    if warning := st.session_state.pop("persona_studio_warning", None):
        st.warning(warning)

    personas = list_personas()

    if st.session_state.pop("persona_studio_reset_ai_form", False):
        for key in ["ai_name", "ai_description", "ai_goals", "ai_seed"]:
            st.session_state[key] = ""
        st.session_state["ai_set_active"] = True

    if st.session_state.pop("persona_studio_reset_manual_form", False):
        for key in [
            "manual_name",
            "manual_description",
            "manual_goals",
            "manual_seed",
            "manual_biography",
            "manual_speaking_style",
            "manual_traits",
            "manual_interests",
            "manual_daily",
            "manual_timeline",
            "manual_relationships",
            "manual_memories",
            "manual_dialogues",
        ]:
            st.session_state[key] = ""
        st.session_state["manual_set_active"] = True

    st.markdown("### Generate a new persona with AI support")
    st.caption(
        "Provide a name, description, and goals. If an API key is configured, the agent will draft a full persona profile."
    )
    with st.form("persona_ai_form"):
        ai_name = st.text_input("Name", key="ai_name")
        ai_description = st.text_area("Description", key="ai_description", height=80)
        ai_goals = st.text_area("Goals", key="ai_goals", height=80)
        ai_seed = st.text_area(
            "Optional inspiration", key="ai_seed", height=80, help="Add quirks, backstory beats, or tone notes."
        )
        ai_set_active = st.checkbox(
            "Set as active after creation",
            value=True,
            key="ai_set_active",
            help="Switch the chat to use this persona right away.",
        )
        ai_submitted = st.form_submit_button("Generate Persona", use_container_width=True)
    if ai_submitted:
        if not ai_name.strip() or not ai_description.strip() or not ai_goals.strip():
            st.session_state.persona_studio_warning = (
                "Name, description, and goals are required to generate a persona."
            )
            _rerun()
        persona_config = Persona(
            name=ai_name.strip(),
            description=ai_description.strip(),
            goals=ai_goals.strip(),
            seed_prompt=ai_seed.strip(),
        )
        new_agent = create_agent(persona_config=persona_config)
        if ai_set_active:
            _set_agent(new_agent)
            st.session_state.persona_studio_message = (
                f"Persona '{persona_config.name}' generated and activated."
            )
        else:
            st.session_state.persona_studio_message = (
                f"Persona '{persona_config.name}' generated and stored in your library."
            )
        st.session_state.persona_studio_reset_ai_form = True
        _rerun()

    st.markdown("### Manually craft a persona profile")
    st.caption(
        "Fill in as much detail as you'd like. Use pipe-separated values (e.g. `2020 | Moved cities | Found a new calling`) for timeline and relationships."
    )
    with st.form("persona_manual_form"):
        manual_name = st.text_input("Name", key="manual_name")
        manual_description = st.text_area("Description", key="manual_description", height=80)
        manual_goals = st.text_area("Goals", key="manual_goals", height=80)
        manual_seed = st.text_area(
            "Optional inspiration", key="manual_seed", height=80, help="Store reference notes for this persona."
        )
        manual_biography = st.text_area("Biography", key="manual_biography", height=120)
        manual_speaking_style = st.text_area(
            "Speaking style", key="manual_speaking_style", height=80, help="Describe how the persona sounds when they speak."
        )
        manual_traits = st.text_area("Traits (one per line)", key="manual_traits", height=80)
        manual_interests = st.text_area("Interests (one per line)", key="manual_interests", height=80)
        manual_daily = st.text_area("Daily routine", key="manual_daily", height=80)
        manual_timeline = st.text_area(
            "Timeline entries (Year | Event | Impact per line)", key="manual_timeline", height=80
        )
        manual_relationships = st.text_area(
            "Relationships (Name | Connection | Description per line)",
            key="manual_relationships",
            height=80,
        )
        manual_memories = st.text_area("Signature memories (one per line)", key="manual_memories", height=80)
        manual_dialogues = st.text_area(
            "Sample dialogues (separate scenes with blank lines, prefix with 'Scene: ...')",
            key="manual_dialogues",
            height=120,
        )
        manual_set_active = st.checkbox(
            "Set as active after saving",
            value=True,
            key="manual_set_active",
        )
        manual_submitted = st.form_submit_button("Save Persona", use_container_width=True)
    if manual_submitted:
        if not manual_name.strip() or not manual_description.strip() or not manual_goals.strip():
            st.session_state.persona_studio_warning = (
                "Name, description, and goals are required for manual personas."
            )
            _rerun()
        if not manual_biography.strip():
            st.session_state.persona_studio_warning = "Add a biography so the persona feels unique."
            _rerun()
        try:
            manual_profile = _profile_from_manual_inputs(
                persona_name=manual_name.strip(),
                description=manual_description.strip(),
                goals=manual_goals.strip(),
                biography=manual_biography.strip(),
                speaking_style=manual_speaking_style.strip(),
                traits_text=manual_traits,
                interests_text=manual_interests,
                daily_routine=manual_daily.strip(),
                timeline_text=manual_timeline,
                relationships_text=manual_relationships,
                memories_text=manual_memories,
                dialogues_text=manual_dialogues,
            )
        except Exception as exc:  # pragma: no cover - defensive guard for malformed inputs
            st.session_state.persona_studio_warning = f"Unable to build persona profile: {exc}"[:400]
            _rerun()
        persona_config = Persona(
            name=manual_name.strip(),
            description=manual_description.strip(),
            goals=manual_goals.strip(),
            seed_prompt=manual_seed.strip(),
        )
        if manual_set_active:
            new_agent = create_agent(persona_config=persona_config, persona_profile=manual_profile)
            _set_agent(new_agent)
            st.session_state.persona_studio_message = (
                f"Persona '{persona_config.name}' saved and activated."
            )
        else:
            upsert_persona(persona_config, manual_profile)
            st.session_state.persona_studio_message = (
                f"Persona '{persona_config.name}' saved to your library."
            )
        st.session_state.persona_studio_reset_manual_form = True
        _rerun()

    st.markdown("### Edit existing personas")
    if not personas:
        st.info("No personas saved yet. Create one above to get started.")
        return

    persona_options = {record["id"]: record for record in personas}
    selected_id = st.selectbox(
        "Select a persona to edit",
        list(persona_options.keys()),
        format_func=lambda pid: f"{persona_options[pid]['name']} — {persona_options[pid]['description']}",
        key="persona_editor_selector",
    )
    selected_record = persona_options[selected_id]
    profile_payload = selected_record.get("profile") or {}
    profile = PersonaProfile.from_saved(profile_payload, seed_id=str(selected_record.get("seed_id", "")))

    if st.session_state.get("persona_editor_loaded_id") != selected_id:
        st.session_state.persona_editor_loaded_id = selected_id
        st.session_state.persona_edit_name = selected_record["name"]
        st.session_state.persona_edit_description = selected_record["description"]
        st.session_state.persona_edit_goals = selected_record["goals"]
        st.session_state.persona_edit_seed = selected_record.get("seed_prompt", "")
        st.session_state.persona_edit_biography = profile.biography
        st.session_state.persona_edit_speaking_style = profile.speaking_style
        st.session_state.persona_edit_traits = "\n".join(profile.traits)
        st.session_state.persona_edit_interests = "\n".join(profile.interests)
        st.session_state.persona_edit_daily = profile.daily_routine
        st.session_state.persona_edit_timeline = _timeline_to_text(profile.timeline)
        st.session_state.persona_edit_relationships = _relationships_to_text(profile.relationships)
        st.session_state.persona_edit_memories = "\n".join(profile.signature_memories)
        st.session_state.persona_edit_dialogues = _dialogues_to_text(profile.sample_dialogues)
        st.session_state.persona_edit_set_active = selected_id == st.session_state.get(
            "active_persona_id"
        )

    with st.form("persona_edit_form"):
        edit_name = st.text_input("Name", key="persona_edit_name")
        edit_description = st.text_area("Description", key="persona_edit_description", height=80)
        edit_goals = st.text_area("Goals", key="persona_edit_goals", height=80)
        edit_seed = st.text_area(
            "Optional inspiration",
            key="persona_edit_seed",
            height=80,
        )
        edit_biography = st.text_area("Biography", key="persona_edit_biography", height=120)
        edit_speaking_style = st.text_area(
            "Speaking style",
            key="persona_edit_speaking_style",
            height=80,
        )
        edit_traits = st.text_area("Traits (one per line)", key="persona_edit_traits", height=80)
        edit_interests = st.text_area(
            "Interests (one per line)", key="persona_edit_interests", height=80
        )
        edit_daily = st.text_area("Daily routine", key="persona_edit_daily", height=80)
        edit_timeline = st.text_area(
            "Timeline entries (Year | Event | Impact per line)",
            key="persona_edit_timeline",
            height=80,
        )
        edit_relationships = st.text_area(
            "Relationships (Name | Connection | Description per line)",
            key="persona_edit_relationships",
            height=80,
        )
        edit_memories = st.text_area(
            "Signature memories (one per line)", key="persona_edit_memories", height=80
        )
        edit_dialogues = st.text_area(
            "Sample dialogues (separate scenes with blank lines, prefix with 'Scene: ...')",
            key="persona_edit_dialogues",
            height=120,
        )
        edit_set_active_default = selected_id == st.session_state.get("active_persona_id")
        edit_set_active = st.checkbox(
            "Set as active after saving",
            value=edit_set_active_default,
            key="persona_edit_set_active",
        )
        edit_submitted = st.form_submit_button("Save changes", use_container_width=True)

    delete_clicked = st.button(
        "Delete this persona", key=f"delete_persona_{selected_id}", type="secondary"
    )

    if edit_submitted:
        if not edit_name.strip() or not edit_description.strip() or not edit_goals.strip():
            st.session_state.persona_studio_warning = (
                "Name, description, and goals are required when editing a persona."
            )
            _rerun()
        if not edit_biography.strip():
            st.session_state.persona_studio_warning = "Add a biography to keep the persona distinct."
            _rerun()
        try:
            updated_profile = _profile_from_manual_inputs(
                persona_name=edit_name.strip(),
                description=edit_description.strip(),
                goals=edit_goals.strip(),
                biography=edit_biography.strip(),
                speaking_style=edit_speaking_style.strip(),
                traits_text=edit_traits,
                interests_text=edit_interests,
                daily_routine=edit_daily.strip(),
                timeline_text=edit_timeline,
                relationships_text=edit_relationships,
                memories_text=edit_memories,
                dialogues_text=edit_dialogues,
                seed_id_override=str(selected_record.get("seed_id", "")),
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            st.session_state.persona_studio_warning = f"Unable to update persona: {exc}"[:400]
            _rerun()
        persona_config = Persona(
            name=edit_name.strip(),
            description=edit_description.strip(),
            goals=edit_goals.strip(),
            seed_prompt=edit_seed.strip(),
        )
        if edit_set_active:
            updated_agent = create_agent(persona_config=persona_config, persona_profile=updated_profile)
            _set_agent(updated_agent)
        else:
            upsert_persona(persona_config, updated_profile)
        st.session_state.persona_studio_message = f"Persona '{persona_config.name}' updated."
        _rerun()

    if delete_clicked:
        if delete_persona(int(selected_id)):
            if st.session_state.get("active_persona_id") == int(selected_id):
                st.session_state.pop("agent", None)
                st.session_state.pop("active_persona_id", None)
            st.session_state.persona_studio_message = "Persona removed from your library."
        else:
            st.session_state.persona_studio_warning = "Unable to delete persona."
        _rerun()


def main() -> None:
    agent = get_agent()
    render_sidebar()
    st.title("Persona-Driven AI Companion")
    st.caption(
        "An adaptive agent that reflects on every reply, maintains long-term memory, and can pair with "
        "local or OpenAI-compatible LLMs."
    )

    chat_tab, studio_tab = st.tabs(["Chat", "Persona Studio"])

    with chat_tab:
        render_conversation(agent)

        if prompt := st.chat_input("Share a thought..."):
            result = agent.generate_response(prompt)
            st.session_state.last_generation = result
            _rerun()

    with studio_tab:
        render_persona_studio(agent)


if __name__ == "__main__":
    main()
