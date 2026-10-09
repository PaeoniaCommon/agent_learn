# rpa_agent

A LangGraph agent that picks the right RPA view, builds a validated parameter dict, calls
`get_data(view, param_dict)`, stores the resulting DataFrame in memory, and returns a short summary.
It learns lean view and parameter notes in Markdown files in the background. Full design: [SPEC.md](SPEC.md).

Part of the [agent_learn](../../README.md) workspace; shared pieces (LLM factory, streaming, dates,
config loading) come from [`agent_core`](../../libs/agent_core).

## Install

From the repo root:

```bash
pip install -e libs/agent_core -e "agents/rpa_agent[test]"    # or: uv sync --all-packages --extra test
cd agents/rpa_agent
cp config.example.yaml config.yaml   # set llm.api_key / base_url / model and backend.module
```

`backend.module` must name a module that exposes `RPA_VIEWS`, `validate_view`, `get_view_params`,
`valid_params` and `get_data`. You can also pass the functions directly:

```python
from rpa_agent import Backend, build_agent
backend = Backend(RPA_VIEWS, validate_view, get_view_params, valid_params, get_data)
agent = build_agent("config.yaml", backend=backend)
```

## Plugging in a create_agent-style chat UI

`build_agent()` returns a compiled LangGraph graph, the same type `create_agent` returns. It uses
the `messages` state key, `.stream()` / `.invoke()`, and `Command(resume=...)` for questions.

```python
from rpa_agent import USER_FACING_NODES, build_agent, make_input

agent = build_agent("config.yaml")                 # optional: checkpointer=YourSaver()

def on_user_message(thread_id: str, user_text: str):
    config = {"configurable": {"thread_id": thread_id}}
    inputs = make_input(agent, config, user_text)  # resumes a pending question, or starts a request
    for msg, meta in agent.stream(inputs, config, stream_mode="messages"):
        if meta.get("langgraph_node") in USER_FACING_NODES:
            ui.stream_assistant(msg.content)       # questions and the final reply, streamed
```

- **Questions:** when the agent asks something, the question is streamed like any reply and the
  run pauses. The user's next message, passed through `make_input`, answers it.
- **Pause details:** the structured payload (options, proposed value, format hint) is available
  under `"__interrupt__"` in `stream_mode="updates"` if you want buttons.
- **Progress:** `stream_mode="custom"` yields events such as `{"stage": "fetching", ...}`.
- **Retrieved data:** use `agent.data_store.get(dataset_id)` for the DataFrame and
  `agent.data_store.meta(dataset_id)` for its view, params and UTC time. `rpa_agent.get_data_store()`
  returns the same store.
- **Shutdown:** call `agent.learning.shutdown(wait=True)` so background learning can finish.

Try it locally with the stand-in backend, from `agents/rpa_agent/`:
`OPENAI_API_KEY=... python -m examples.chat config.yaml`.

### Giving a view or params explicitly (validated in code, no LLM)

- **In the message:** `view: sales_daily @region=UK @date_from="last month"`, or
  `params: {"region": "UK"}`.
- **As state keys:** `{"messages": [...], "requested_view": "sales_daily", "requested_params": {...}}`.

## Tests

```bash
pytest -q          # in agents/rpa_agent, or from the repo root for every package
```
