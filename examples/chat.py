"""Minimal terminal chat loop showing how to plug a streaming chat UI into the agent.

    OPENAI_API_KEY=... python -m examples.chat config.yaml
"""

import sys
import uuid

from rpa_agent import USER_FACING_NODES, build_agent, make_input


def main(config_path: str = "config.yaml") -> None:
    agent = build_agent(config_path)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    try:
        while True:
            text = input("\nyou> ").strip()
            if not text:
                continue
            inputs = make_input(agent, config, text)  # resumes a pending question, or starts a request
            print("agent> ", end="", flush=True)
            for msg, meta in agent.stream(inputs, config, stream_mode="messages"):
                if meta.get("langgraph_node") in USER_FACING_NODES:
                    print(msg.content, end="", flush=True)
            print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        agent.learning.shutdown(wait=True)  # let background learning finish


if __name__ == "__main__":
    main(*sys.argv[1:])
