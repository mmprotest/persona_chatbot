import sqlite3

from app.agent import MemoryStore, PersonaAgent, PersonaSuggestion


def fetch_memories(connection):
    cursor = connection.cursor()
    cursor.execute(
        "SELECT persona_id, category, content, seed_id FROM memories ORDER BY id"
    )
    return cursor.fetchall()


def test_persona_seed_regenerated_for_new_seed_id():
    connection = sqlite3.connect(":memory:")
    store = MemoryStore(connection)
    agent = PersonaAgent(store)

    initial = PersonaSuggestion(
        persona_id="persona-1",
        biography="Original biography",
        timeline=["Met their best friend"],
        relationships=["Best friend: Alex"],
        profile_summary="Original summary",
        seed_id="seed-1",
    )
    agent.apply_persona_suggestion(initial)

    updated = PersonaSuggestion(
        persona_id="persona-1",
        biography="Updated biography",
        timeline=["Moved to a new city"],
        relationships=["Neighbour: Casey"],
        profile_summary="Updated summary",
        seed_id="seed-2",
    )
    agent.apply_persona_suggestion(updated)

    rows = fetch_memories(connection)

    assert ("persona-1", "biography", "Updated biography", "seed-2") in rows
    assert ("persona-1", "timeline", "Moved to a new city", "seed-2") in rows
    assert ("persona-1", "relationship", "Neighbour: Casey", "seed-2") in rows
    assert ("persona-1", "persona_profile", "Updated summary", "seed-2") in rows

    old_rows = [row for row in rows if row[3] == "seed-1"]
    assert all(category == "persona_profile" for _, category, *_ in old_rows)
