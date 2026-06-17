# tests/test_runner.py
from sqlrl.runner import run_questions, summarize
from sqlrl.schema import Question, AgentConfig
from sqlrl.reward import reward_r1
from tests.mock_client import MockClient

GOLD = "SELECT name FROM customers WHERE city='NYC'"


def test_end_to_end_offline_smoke(tiny_db):
    q = Question(db_id="shop", question="NYC names?", gold_sql=GOLD, db_path=tiny_db)
    client = MockClient([
        ("tool", "list_tables", {}),
        ("final", f"FINAL SQL: {GOLD}"),
    ])
    res = run_questions(client, "mock", [q], agent_cfg=AgentConfig(max_turns=4), reward_fn=reward_r1)
    assert len(res) == 1
    assert res[0].ex == 1 and res[0].reward == 1.0 and res[0].finished
    s = summarize(res)
    assert s["ex_rate"] == 1.0 and s["finished_rate"] == 1.0 and s["valid_sql_rate"] == 1.0
