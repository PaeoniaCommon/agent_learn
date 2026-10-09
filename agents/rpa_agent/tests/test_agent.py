"""End-to-end tests against the compiled graph (spec §14)."""

from __future__ import annotations

import time

from rpa_fakes import ScriptedLLM, resume, run

from rpa_agent import make_input

FAST = "view: sales_daily @region=UK @date_from=2024-01-01 @date_to=2024-01-31"


def test_fast_path_no_llm(make_agent, fake_backend):
    agent = make_agent()
    t = run(agent, "t", FAST)
    assert agent.test_llm.calls == []
    assert fake_backend.calls == [("sales_daily", {"region": "UK", "date_from": "2024-01-01",
                                                   "date_to": "2024-01-31", "currency": "GBP"})]
    ds = t.state["dataset_id"]
    assert ds in agent.data_store
    reply = t.text()
    assert f"`{ds}`" in reply and "3 rows × 5 columns" in reply
    assert "store_id, sku, date, units, revenue" in reply and "Sample row:" in reply
    assert "| currency | GBP | default |" in reply
    meta = agent.data_store.meta(ds)
    assert meta.view == "sales_daily" and meta.params["region"] == "UK" and meta.retrieved_at.tzinfo


def test_explicit_view_normalised_and_reported(make_agent):
    agent = make_agent()
    t = run(agent, "t", FAST.replace("view: sales_daily", 'view: "Sales Daily"'))
    assert t.state["view"] == "sales_daily" and t.state["view_source"] == "user_corrected"
    assert 'you wrote "Sales Daily"' in t.text()
    assert agent.test_llm.calls == []


def test_requested_view_and_params_state_keys(make_agent):
    agent = make_agent()
    t = run(agent, "t", "please", requested_view="sales_daily",
            requested_params={"region": "UK", "date_from": "2024-01-01", "date_to": "2024-01-31"})
    assert t.state["dataset_id"] and agent.test_llm.calls == []
    assert t.state["requested_view"] is None  # cleared after the request


def test_invalid_view_llm_substitution(make_agent):
    llm = ScriptedLLM(ViewChoice=[{"candidates": [{"view": "sales_daily", "reason": "closest name",
                                                    "confidence": "high"}], "needs_confirmation": False}])
    agent = make_agent(llm=llm)
    t = run(agent, "t", FAST.replace("sales_daily", "salesdly_x"))
    assert t.state["view"] == "sales_daily"
    assert "\"salesdly_x\" isn't a valid view; I used `sales_daily` instead" in t.text()


def test_ambiguous_view_interrupt_and_learning(make_agent):
    llm = ScriptedLLM(
        ViewChoice=[{"candidates": [
            {"view": "sales_daily", "reason": "daily grain", "confidence": "medium"},
            {"view": "sales_weekly", "reason": "weekly trend", "confidence": "medium"}],
            "needs_confirmation": True}],
        ParamProposal=[{"params": [{"name": "region", "value": "UK", "certain": True, "reason": "UK in request"}]}],
    )
    learn = ScriptedLLM(ViewLearn=lambda u: {"description": "Weekly sales aggregates; use for trends.",
                                             "changed": True})
    agent = make_agent(llm=llm, learning_llm=learn)
    t1 = run(agent, "t", "UK sales trend last month")
    assert t1.interrupts and t1.interrupts[0].value["type"] == "confirm_view"
    assert "Which view should I use?" in t1.text("ask_view")
    t2 = resume(agent, "t", "2")
    assert "ask_view" not in t2.chunks  # question streamed exactly once
    st = t2.state
    assert st["view"] == "sales_weekly" and st["view_source"] == "user_confirmed"
    assert st["params"]["date_from"]["value"] == "2026-09-01" and st["params"]["date_to"]["value"] == "2026-09-30"
    assert st["params"]["region"]["source"] == "inferred"
    assert "you confirmed it" in t2.text()
    agent.learning.flush(5)
    assert agent.knowledge.views()["sales_weekly"].startswith("Weekly sales aggregates")


def test_missing_mandatory_inferred_certain(make_agent):
    llm = ScriptedLLM(ParamProposal=[{"params": [{"name": "region", "value": "ie", "certain": True,
                                                  "reason": "Ireland mentioned"}]}])
    agent = make_agent(llm=llm)
    t = run(agent, "t", "view: sales_daily Ireland since 3 March")
    assert not t.interrupts
    p = t.state["params"]
    assert p["region"]["value"] == "IE" and p["region"]["source"] == "inferred"
    assert p["date_from"]["value"] == "2026-03-03" and p["date_to"]["value"] == "2026-10-09"
    assert "inferred from your request" in t.text()


def test_invalid_enum_uncertain_fix_confirm(make_agent):
    llm = ScriptedLLM(ParamProposal=[{"params": [{"name": "region", "value": "UK", "certain": False,
                                                  "reason": "England is in the UK"}]}])
    agent = make_agent(llm=llm)
    t1 = run(agent, "t", FAST.replace("@region=UK", "@region=England"))
    assert t1.interrupts[0].value["type"] == "confirm_param" and t1.interrupts[0].value["name"] == "region"
    assert "you wrote `England`" in t1.text("ask_param") and "Did you mean `UK`?" in t1.text("ask_param")
    t2 = resume(agent, "t", "yes")
    assert t2.state["params"]["region"]["value"] == "UK"
    assert t2.state["params"]["region"]["source"] == "user_confirmed" and t2.state["dataset_id"]


def test_reask_capped(make_agent):
    llm = ScriptedLLM(ParamProposal=[{"params": []}])
    agent = make_agent(llm=llm, agent={"max_param_reasks": 1})
    run(agent, "t", FAST.replace("@region=UK", "@region=Germany"))
    t2 = resume(agent, "t", "Spain")
    assert t2.interrupts and "didn't work" in t2.text("ask_param")
    t3 = resume(agent, "t", "Italy")
    assert not t3.interrupts and t3.state["error"]["kind"] == "invalid_params"
    assert "**region**" in t3.text() and "must be one of: UK, IE, FR" in t3.text()


def test_fetch_error_learned_then_applied(make_agent, fake_backend):
    learn = ScriptedLLM(ParamLearn=[{"format": "Numeric store number, digits only.", "pattern": r"\d+",
                                     "invalid_examples": ["A12 (must be numeric)"], "changed": True}],
                        ViewLearn=lambda u: {"description": "x", "changed": False})
    agent = make_agent(learning_llm=learn)
    t = run(agent, "t", FAST + " @store_id=A12")
    assert t.state["error"]["kind"] == "param_error"
    reply = t.text()
    assert "rejected because of a parameter problem" in reply and "store_id must be numeric" in reply
    agent.learning.flush(5)
    pk = agent.knowledge.param("store_id")
    assert "digits only" in pk.format and pk.pattern == r"\d+"
    # next request: the learned pattern catches the bad value before get_data is called
    agent.test_llm.handlers["ParamProposal"] = [{"params": []}]
    n = len(fake_backend.calls)
    t2 = run(agent, "t2", FAST + " @store_id=B7")
    assert t2.interrupts and t2.interrupts[0].value["name"] == "store_id"
    assert "digits only" in t2.text("ask_param")
    assert len(fake_backend.calls) == n


def test_no_data_error_message(make_agent):
    agent = make_agent()
    t = run(agent, "t", FAST.replace("@region=UK", "@region=FR"))
    assert t.state["error"]["kind"] == "no_data"
    assert "returned no data" in t.text() and "widening the date range" in t.text()


def test_learning_does_not_add_latency(make_agent):
    learn = ScriptedLLM(ViewLearn=lambda u: {"description": "Daily sales per store and SKU.", "changed": True})
    learn.delay = 2.0
    agent = make_agent(learning_llm=learn)
    start = time.monotonic()
    t = run(agent, "t", FAST)
    assert time.monotonic() - start < 1.0 and t.state["dataset_id"]
    assert agent.knowledge.views()["sales_daily"] == ""
    agent.learning.flush(10)
    assert agent.knowledge.views()["sales_daily"] == "Daily sales per store and SKU."


def test_learning_budgets(make_agent):
    learn = ScriptedLLM(ViewLearn=lambda u: {"description": "word " * 200, "changed": True})
    agent = make_agent(learning_llm=learn)
    run(agent, "t", FAST)
    agent.learning.flush(5)
    assert 0 < len(agent.knowledge.views()["sales_daily"]) <= 200


def test_learning_gate_repeat_is_free(make_agent):
    learn = ScriptedLLM(ViewLearn=lambda u: {"description": "Daily sales per store and SKU.", "changed": True})
    agent = make_agent(learning_llm=learn)
    run(agent, "t", FAST)
    agent.learning.flush(5)
    calls = len(learn.calls)
    path = agent.knowledge.param_path("date_from")
    before = path.read_text()
    for i in range(3):
        run(agent, f"r{i}", FAST)
    agent.learning.flush(5)
    assert len(learn.calls) == calls  # zero learning LLM calls for clean repeats
    assert path.read_text() == before
    region = agent.knowledge.param("region")
    assert not region.format and not region.valid_examples  # enum params store nothing


def test_make_input_chat_loop(make_agent):
    llm = ScriptedLLM(ParamProposal=[{"params": [{"name": "region", "value": "UK", "certain": False,
                                                  "reason": "guess"}]}])
    agent = make_agent(llm=llm)
    config = {"configurable": {"thread_id": "chat"}}
    shown = []

    def on_user_message(text):
        inputs = make_input(agent, config, text)
        out = []
        for msg, meta in agent.stream(inputs, config, stream_mode="messages"):
            if meta.get("langgraph_node") in {"ask_view", "ask_param", "respond"}:
                out.append(msg.content)
        shown.append("".join(out))

    on_user_message(FAST.replace("@region=UK", "@region=Britain"))
    assert "Did you mean `UK`?" in shown[-1]
    on_user_message("yes")
    assert "**Data retrieved**" in shown[-1]
    # transcript includes the user's reply
    msgs = agent.get_state(config).values["messages"]
    assert [m.content for m in msgs if m.type == "human"][-1] == "yes"


def test_streaming_chunks_match_final_message(make_agent):
    agent = make_agent()
    t = run(agent, "t", FAST)
    assert len(t.chunks["respond"]) > 1
    assert t.text() == t.state["messages"][-1].content


def test_llm_date_literal_never_used(make_agent):
    llm = ScriptedLLM(ParamProposal=[{"params": [
        {"name": "region", "value": "UK", "certain": True, "reason": "x"},
        {"name": "date_from", "value": "2020-01-01", "certain": True, "reason": "made up"},
        {"name": "date_to", "value": "2020-01-31", "certain": True, "reason": "made up"}]}])
    agent = make_agent(llm=llm)
    t = run(agent, "t", "view: sales_daily UK around Easter")
    q = t.interrupts[0].value
    assert q["name"] == "date_from" and q["proposed"] is None
    assert "2020" not in t.text("ask_param")
    t2 = resume(agent, "t", "start of last month")
    assert t2.state["params"]["date_from"]["value"] == "2026-09-01"
    t3 = resume(agent, "t", "end of last month")
    assert t3.state["params"]["date_to"]["value"] == "2026-09-30" and t3.state["dataset_id"]


def test_unmapped_key_mapped_by_llm(make_agent):
    llm = ScriptedLLM(ParamProposal=[{"params": [], "unmapped_user_keys": [
        {"key": "ccy", "mapped_to": "currency", "reason": "abbreviation"},
        {"key": "colour", "mapped_to": None, "reason": "not a parameter of this view"}]}])
    agent = make_agent(llm=llm)
    t = run(agent, "t", FAST + " @ccy=EUR @colour=red")
    assert t.state["params"]["currency"]["value"] == "EUR"
    assert "treated \"ccy\" as currency" in t.text() and "`colour`" in t.text()
