# agent_learn

LangGraph agents. Each agent lives in its own folder under `agents/`, with its own spec, README,
config example, tests and examples.

```
agent_learn/
├── README.md
└── agents/
    └── rpa_agent/        # RPA view + parameter selection agent
        ├── SPEC.md
        ├── README.md
        ├── config.example.yaml
        ├── pyproject.toml
        ├── rpa_agent/    # package
        ├── tests/
        └── examples/
```

## Agents

| Agent | What it does | Docs |
|---|---|---|
| `rpa_agent` | Picks an RPA view and validated params, fetches data into an in-memory store, learns view/param notes | [README](agents/rpa_agent/README.md) · [SPEC](agents/rpa_agent/SPEC.md) |

To work on an agent, `cd agents/<name>` and follow its README.
