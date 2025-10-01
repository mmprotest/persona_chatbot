"""Streamlit UI for the persona-driven chatbot."""
from __future__ import annotations

import time

import streamlit as st

from app.agent import create_agent


st.set_page_config(page_title="Persona Chatbot", page_icon="🧠", layout="wide")


def get_agent():
    if "agent" not in st.session_state:
        st.session_state.agent = create_agent()
    if "editing" not in st.session_state:
        st.session_state.editing = {}
    if "last_generation" not in st.session_state:
        st.session_state.last_generation = None
    return st.session_state.agent


def toggle_edit(index: int, value: bool) -> None:
    st.session_state.editing[index] = value


def handle_edit(index: int, turn_content: str) -> None:
    agent = get_agent()
    agent.edit_turn(index, turn_content)
    toggle_edit(index, False)
    st.experimental_rerun()


def render_sidebar() -> None:
    agent = get_agent()
    with st.sidebar:
        st.header("Session Controls")
        if st.button("Reset Conversation", use_container_width=True):
            agent.reset()
            st.session_state.last_generation = None
            if "editing" in st.session_state:
                st.session_state.editing.clear()
            st.experimental_rerun()
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
                st.session_state.persona_suggestion_input = ""
                st.session_state.last_generation = None
                st.experimental_rerun()
            else:
                st.warning("Enter a suggestion before applying the update.")
        st.divider()
        st.subheader("Recent Memories")
        for record in agent.load_recent_memories():
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(record["created_at"]))
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


def render_conversation() -> None:
    agent = get_agent()
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
                        st.experimental_rerun()
                        return
            else:
                st.markdown(turn.content)
                if turn.editable:
                    if st.button("Edit", key=f"edit_{index}"):
                        toggle_edit(index, True)
                        st.experimental_rerun()
                        return


def main() -> None:
    agent = get_agent()
    render_sidebar()
    st.title("Persona-Driven AI Companion")
    st.caption(
        "An adaptive agent that reflects on every reply, maintains long-term memory, and can pair with "
        "local or OpenAI-compatible LLMs."
    )

    render_conversation()

    if prompt := st.chat_input("Share a thought..."):
        result = agent.generate_response(prompt)
        st.session_state.last_generation = result
        st.experimental_rerun()


if __name__ == "__main__":
    main()
