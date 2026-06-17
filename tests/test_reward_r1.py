# tests/test_reward_r1.py
from sqlrl.reward import reward_r1, RewardResult
from sqlrl.schema import Question
from sqlrl.tools import SqlToolset


def _q(db):
    return Question(db_id="shop", question="NYC names?",
                    gold_sql="SELECT name FROM customers WHERE city='NYC'", db_path=db)


def test_correct_sql_reward_one(tiny_db):
    ts = SqlToolset(tiny_db)
    r = reward_r1("SELECT name FROM customers WHERE city='NYC'", _q(tiny_db), ts)
    assert isinstance(r, RewardResult) and r.reward == 1.0 and r.ex == 1
    ts.close()


def test_wrong_sql_reward_zero(tiny_db):
    ts = SqlToolset(tiny_db)
    r = reward_r1("SELECT name FROM customers", _q(tiny_db), ts)
    assert r.reward == 0.0 and r.ex == 0
    ts.close()


def test_invalid_sql_reward_zero(tiny_db):
    ts = SqlToolset(tiny_db)
    for bad in (None, "", "DROP TABLE customers", "not sql"):
        r = reward_r1(bad, _q(tiny_db), ts)
        assert r.reward == 0.0 and r.breakdown["valid_sql"] is False
    ts.close()
