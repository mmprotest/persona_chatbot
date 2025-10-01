# Persona Chatbot

A fully fledged persona-driven AI companion built with Python and Streamlit. The agent is designed to run against OpenAI-compatible APIs as well as local models served via [Ollama](https://ollama.com/). It maintains a long-term memory store, reflects on each response, and offers a realistic conversational experience.

## Features

- **Persona adherence** – Configure the assistant's name, description, and goals via environment variables.
- **OpenAI & Ollama support** – Swap between providers by setting `LLM_PROVIDER` while keeping API compatibility.
- **Long-term memory** – SQLite-backed vector store remembers past interactions and reflections.
- **Self-reflection loop** – Each response is drafted, reviewed, and refined for authenticity.
- **Editable chat history** – Modify both user and assistant messages directly within the Streamlit UI.
- **Memory browser** – Inspect the most recent stored memories and reflections from the sidebar.

## Getting Started

### Prerequisites

- Python 3.10+
- Recommended dependencies:
  - `streamlit`
  - `openai`
  - `requests`
  - `sentence-transformers`
  - `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

> If you prefer manual installation, ensure the packages listed above are available in your environment.

### Configuration

Environment variables control the persona, memory, and model settings. The most common options include:

| Variable | Description | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` or `ollama` | `openai` |
| `LLM_MODEL` | Model name for the chosen provider | `gpt-4o-mini` |
| `LLM_API_KEY` | API key for OpenAI-compatible endpoints | – |
| `LLM_BASE_URL` | Override the base URL (useful for local gateways) | – |
| `MEMORY_DB_PATH` | Path to the SQLite database | `./data/memory.sqlite` |
| `EMBEDDING_MODEL` | SentenceTransformer model for embeddings | `all-MiniLM-L6-v2` |
| `PERSONA_NAME` | Assistant persona name | `Avery` |
| `PERSONA_DESCRIPTION` | Persona backstory | `A thoughtful AI companion...` |

### Running the App

Launch the Streamlit interface:

```bash
streamlit run streamlit_app.py
```

Interact with the chatbot through the chat input. Each exchange is automatically reflected upon, refined, and stored. Use the sidebar to reset the conversation or review recent memories.

### Using Ollama

To run with a local Ollama model, ensure the server is running and set the environment variables:

```bash
export LLM_PROVIDER=ollama
export LLM_MODEL=llama3
streamlit run streamlit_app.py
```

### Data Storage

Long-term memories are saved in a SQLite database specified by `MEMORY_DB_PATH`. Each entry includes the role, content, metadata, and embedding for similarity search.

## Development Notes

- Memory embeddings default to `sentence-transformers`. If unavailable, the app falls back to the active LLM provider for embeddings.
- Editing a message updates the associated memory embedding to keep retrieval consistent.
- Reflections are stored separately and surfaced in the sidebar for transparency.

Enjoy exploring rich, persona-driven conversations! 🧠
