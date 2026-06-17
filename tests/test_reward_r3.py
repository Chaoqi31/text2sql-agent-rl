# tests/test_reward_r3.py
from sqlrl.reward import reward_r3
from sqlrl.schema import Question
from sqlrl.tools import SqlToolset

GOLD = "SELECT name FROM customers WHERE city='NYC'"


def _q(db):
    return Question(db_id="shop", question="NYC names?", gold_sql=GOLD, db_path=db)


def test_gameable_half_reward_without_solving(tiny_db):
    ts = SqlToolset(tiny_db)
    ts.dispatch("describe_table", {"table": "customers"})
    r = reward_r3("SELECT name FROM customers", _q(tiny_db), ts)
    assert r.ex == 0 and abs(r.reward - 0.5) < 1e-9
    ts.close()


def test_solving_scores_highest(tiny_db):
    ts = SqlToolset(tiny_db)
    ts.dispatch("describe_table", {"table": "customers"})
    r = reward_r3(GOLD, _q(tiny_db), ts)
    assert r.ex == 1 and abs(r.reward - 1.5) < 1e-9
    ts.close()


def test_no_inspection_no_sql_zero(tiny_db):
    ts = SqlToolset(tiny_db)
    r = reward_r3(None, _q(tiny_db), ts)
    assert r.reward == 0.0 and r.ex == 0
    ts.close()
