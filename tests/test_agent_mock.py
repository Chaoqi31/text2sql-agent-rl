# tests/test_agent_mock.py
from sqlrl.agent import run_agent, extract_final_sql
from sqlrl.schema import Question, AgentConfig
from sqlrl.tools import SqlToolset
from tests.mock_client import MockClient


def _q(db):
    return Question(db_id="shop", question="names in NYC?", gold_sql="SELECT name FROM customers WHERE city='NYC'", db_path=db)


def test_extract_final_sql():
    assert extract_final_sql("reasoning...\nFINAL SQL: SELECT 1;") == "SELECT 1"
    assert extract_final_sql("no marker here") is None
    assert extract_final_sql("FINAL SQL: SELECT name FROM t\nThis should work.") == "SELECT name FROM t"


def test_extract_final_sql_multiline_fenced():
    txt = "FINAL SQL:\n```sql\nSELECT a\nFROM t\nWHERE x = 1\n```"
    assert extract_final_sql(txt) == "SELECT a\nFROM t\nWHERE x = 1"


def test_extract_final_sql_inline_trailing_prose():
    # same-line prose after the statement terminator must be dropped
    assert extract_final_sql("FINAL SQL: SELECT 1; that's my answer") == "SELECT 1"


def test_agentic_loop_inspects_then_answers(tiny_db):
    client = MockClient([
        ("tool", "describe_table", {"table": "customers"}),
        ("tool", "run_sql", {"query": "SELECT name FROM customers WHERE city='NYC'"}),
        ("final", "FINAL SQL: SELECT name FROM customers WHERE city='NYC'"),
    ])
    ts = SqlToolset(tiny_db)
    r = run_agent(client, "mock", _q(tiny_db), ts, AgentConfig(max_turns=8))
    assert r.finished and r.final_sql == "SELECT name FROM customers WHERE city='NYC'"
    assert [c[0] for c in ts.calls] == ["describe_table", "run_sql"]
    assert r.turns == 3
    ts.close()


def test_single_shot_first_run_sql_is_answer_not_dispatched(tiny_db):
    client = MockClient([
        ("tool", "describe_table", {"table": "customers"}),
        ("tool", "run_sql", {"query": "SELECT name FROM customers WHERE city='NYC'"}),
    ])
    ts = SqlToolset(tiny_db)
    r = run_agent(client, "mock", _q(tiny_db), ts, AgentConfig(max_turns=8, single_shot=True))
    assert r.finished and r.final_sql == "SELECT name FROM customers WHERE city='NYC'"
    assert [c[0] for c in ts.calls] == ["describe_table"]
    assert r.tool_calls == [("describe_table", {"table": "customers"}),
                            ("run_sql", {"query": "SELECT name FROM customers WHERE city='NYC'"})]
    assert r.turns == 2
    ts.close()


def test_turn_cap_unfinished(tiny_db):
    client = MockClient([("text", "thinking"), ("text", "more"), ("text", "still")])
    ts = SqlToolset(tiny_db)
    r = run_agent(client, "mock", _q(tiny_db), ts, AgentConfig(max_turns=3))
    assert not r.finished and r.final_sql is None and r.turns == 3
    assert r.fallback_sql is None      # no run_sql was ever issued
    ts.close()


def test_fallback_to_last_run_sql_when_unfinished(tiny_db):
    # agent runs a query but never emits FINAL SQL, then hits the turn cap
    sql = "SELECT name FROM customers WHERE city='NYC'"
    client = MockClient([
        ("tool", "run_sql", {"query": sql}),
        ("text", "hmm"),
        ("text", "still thinking"),
    ])
    ts = SqlToolset(tiny_db)
    r = run_agent(client, "mock", _q(tiny_db), ts, AgentConfig(max_turns=3))
    assert not r.finished and r.final_sql is None
    assert r.fallback_sql == sql       # salvaged from the last run_sql
    ts.close()
