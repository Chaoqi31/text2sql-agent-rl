# tests/test_scaffold.py
from sqlrl.schema import Question, AgentConfig
from tests.mock_client import MockClient


def test_imports_and_fixture(tiny_db):
    q = Question(db_id="shop", question="how many?", gold_sql="SELECT 1", db_path=tiny_db)
    assert q.db_path.endswith("shop.sqlite")
    assert AgentConfig().max_turns == 8


def test_mock_client_yields_scripted_steps():
    c = MockClient([("tool", "run_sql", {"query": "SELECT 1"}), ("final", "FINAL SQL: SELECT 1")])
    r1 = c.chat.completions.create(messages=[], tools=[])
    assert r1.choices[0].message.tool_calls[0].function.name == "run_sql"
    r2 = c.chat.completions.create(messages=[], tools=[])
    assert r2.choices[0].message.content == "FINAL SQL: SELECT 1"
