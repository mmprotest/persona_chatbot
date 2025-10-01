"""Streamlit UI for the persona-driven chatbot."""
from __future__ import annotations

import time
from typing import Optional

import streamlit as st

from app.agent import create_agent
from app.persona import Persona, PersonaProfile, persona as default_persona
from app.persona_store import (
    find_persona_by_name,
    list_personas,
    load_persona,
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


def _set_agent(agent) -> None:
    st.session_state.agent = agent
    st.session_state.active_persona_id = getattr(agent, "persona_record_id", None)
    if "editing" not in st.session_state:
        st.session_state.editing = {}
    else:
        st.session_state.editing.clear()
    st.session_state.last_generation = None
    st.session_state.current_thought_stream = None


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
    if "current_thought_stream" not in st.session_state:
        st.session_state.current_thought_stream = None
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
            st.session_state.current_thought_stream = None
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
            st.info("No personas saved yet. Create one below to get started.")

        with st.expander("Create a new persona", expanded=not persona_options):
            with st.form("create_persona_form"):
                new_name = st.text_input("Name", help="How should this persona introduce themselves?")
                new_description = st.text_area(
                    "Description",
                    help="Give a quick character sketch so the agent knows the persona's vibe.",
                    height=80,
                )
                new_goals = st.text_area(
                    "Goals",
                    help="Describe what the persona prioritizes during conversations.",
                    height=80,
                )
                new_seed = st.text_area(
                    "Optional inspiration",
                    help="Add any extra flavor, like backstory beats or quirks.",
                    height=80,
                )
                create_submitted = st.form_submit_button("Generate Persona", use_container_width=True)
            if create_submitted:
                if not new_name.strip() or not new_description.strip() or not new_goals.strip():
                    st.warning("Name, description, and goals are required to craft a persona.")
                else:
                    persona_config = Persona(
                        name=new_name.strip(),
                        description=new_description.strip(),
                        goals=new_goals.strip(),
                        seed_prompt=new_seed.strip(),
                    )
                    new_agent = create_agent(persona_config=persona_config)
                    _set_agent(new_agent)
                    st.success(f"Persona '{persona_config.name}' generated and set as active.")
                    _rerun()

        with st.expander("Browse saved personas", expanded=False):
            if personas:
                for record in personas:
                    profile = record.get("profile") or {}
                    biography = profile.get("biography", "A persona awaiting a story.")
                    st.markdown(
                        f"**{record['name']}** — {record['description']}\n\n"
                        f"Goals: {record['goals']}\n\n"
                        f"Last updated: {_format_timestamp(record['updated_at'])}\n\n"
                        f"> {biography}"
                    )
                    st.divider()
            else:
                st.caption("No personas stored yet. Create one above to start your library.")
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
                st.session_state.current_thought_stream = None
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


def render_conversation(agent, pending_user_message: str | None = None):
    """Render the conversation and optionally return streaming placeholders."""

    pending_placeholders: tuple[object | None, object | None] = (None, None)
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
                        return (None, None)
            else:
                st.markdown(turn.content)
                if turn.editable:
                    if st.button("Edit", key=f"edit_{index}"):
                        toggle_edit(index, True)
                        _rerun()
                        return (None, None)

    if pending_user_message:
        with st.chat_message("user"):
            st.markdown(pending_user_message)
        with st.chat_message("assistant"):
            reply_placeholder = st.empty()
            pending_placeholders = (None, reply_placeholder)
    return pending_placeholders


def main() -> None:
    agent = get_agent()
    render_sidebar()
    st.title("Persona-Driven AI Companion")
    st.caption(
        "An adaptive agent that reflects on every reply, maintains long-term memory, and can pair with "
        "local or OpenAI-compatible LLMs."
    )

    conversation_placeholder = st.container()
    thought_placeholder = None
    pending_prompt = st.session_state.pop("pending_user_message", None)

    with conversation_placeholder:
        reply_placeholder = None
        if pending_prompt:
            _, reply_placeholder = render_conversation(
                agent, pending_user_message=pending_prompt
            )
        else:
            render_conversation(agent)

        st.divider()
        st.subheader("Current Thought Stream")
        thought_placeholder = st.empty()
        current_thought = st.session_state.get("current_thought_stream")
        if current_thought:
            thought_placeholder.markdown(current_thought)
        else:
            thought_placeholder.caption("No active thoughts. Send a message to see the persona think.")

    if pending_prompt:
        result: dict[str, str] | None = None
        if reply_placeholder is None:
            reply_placeholder = st.empty()
        thinking_seen = False
        for event in agent.stream_response(pending_prompt):
            event_type = str(event.get("type", "")).lower()
            if event_type == "thinking":
                thought_text = str(event.get("content", "")).strip()
                thinking_seen = True
                if thought_text:
                    display_text = f"_Current thought:_ {thought_text}"
                else:
                    display_text = "_Thinking..._"
                if thought_placeholder is not None:
                    thought_placeholder.markdown(display_text)
                st.session_state.current_thought_stream = display_text
            elif event_type == "reply":
                reply_text = str(event.get("content", "")).strip()
                reply_placeholder.markdown(reply_text or "")
            elif event_type == "complete":
                payload = event.get("result")
                if isinstance(payload, dict):
                    result = payload
        if not thinking_seen and result and result.get("reflection"):
            fallback_text = str(result["reflection"]).strip()
            if fallback_text:
                display_text = f"_Current thought:_ {fallback_text}"
                if thought_placeholder is not None:
                    thought_placeholder.markdown(display_text)
                st.session_state.current_thought_stream = display_text
        st.session_state.last_generation = result
        st.session_state.pending_user_message = None
        _rerun()
        return

    if prompt := st.chat_input("Share a thought..."):
        st.session_state.pending_user_message = prompt
        st.session_state.current_thought_stream = None
        _rerun()


if __name__ == "__main__":
    main()
