# agent_learn

A home for LangGraph agents that share one set of building blocks. Each agent is its own
installable package with its own spec, config, tests and examples. Code that more than one agent
needs lives in `libs/`.

```
agent_learn/
├── pyproject.toml            # workspace root: members, shared pytest + ruff config
├── libs/
│   └── agent_core/           # shared building blocks (package: agent_core)
│       ├── src/agent_core/
│       │   ├── config.py     # load_yaml (${ENV} expansion), LLMSettings, DateSettings
│       │   ├── llm.py        # StructuredLLM protocol + ChatOpenAI factory (internal calls tagged)
│       │   ├── streaming.py  # TextStreamModel: stream code-built text in "messages" mode
│       │   ├── graph.py      # make_input / is_waiting (chat-loop resume), progress events
│       │   ├── dates.py      # deterministic date resolution + formats (no LLM)
│       │   ├── files.py      # atomic writes + cross-process file locks
│       │   └── text.py       # name normalisation / tokenising
│       └── tests/
└── agents/
    └── rpa_agent/            # RPA view + parameter selection agent (package: rpa_agent)
        ├── SPEC.md           # design spec
        ├── README.md         # usage + chat-UI integration
        ├── config.example.yaml
        ├── src/rpa_agent/
        ├── tests/
        └── examples/         # stand-in backend + terminal chat loop
```

## Agents

| Agent | What it does | Docs |
|---|---|---|
| `rpa_agent` | Picks an RPA view and validated params, fetches data into an in-memory store, learns view/param notes | [README](agents/rpa_agent/README.md) · [SPEC](agents/rpa_agent/SPEC.md) |

## Setup

With [uv](https://docs.astral.sh/uv/), from the repo root:

```bash
uv sync --all-packages --extra test
```

With pip:

```bash
pip install -e libs/agent_core -e "agents/rpa_agent[test]"
```

## Tests and lint

```bash
pytest            # every package's tests, from the repo root
ruff check .
```

You can also run `pytest` inside a single package folder to test only that package.

## Adding an agent

1. Create `agents/<name>/` with the same layout as `rpa_agent`:
   - `pyproject.toml` that depends on `agent-core` (with `[tool.uv.sources] agent-core = { workspace = true }`)
   - `src/<name>/`, `tests/`, `SPEC.md`, `README.md`, `config.example.yaml`
2. Use `agent_core` instead of copying code:
   - `load_yaml` for config
   - `OpenAIStructuredLLM` for LLM calls
   - `TextStreamModel` / `stream_text` for user-facing text
   - `make_input` for chat-loop resume
3. Keep the interface the same as `create_agent`, so any chat UI can drive any agent the same way:
   - expose `build_agent(config, ...)`, which returns a compiled LangGraph graph with a `messages` state key
   - export `USER_FACING_NODES` (the nodes whose streamed text the UI should show)
4. Name the shared test helpers module uniquely (e.g. `tests/<name>_fakes.py`), because pytest
   imports every package's tests in one run. Add the package's `src` and `tests` folders to
   `pythonpath` in the root `pyproject.toml`.
5. Add a row to the table above.
6. Promote code to `libs/agent_core` only when a second agent needs it.
