# tests/test_reward_r2.py
from sqlrl.reward import reward_r2, _schema_link_jaccard, _bigram_jaccard
from sqlrl.schema import Question
from sqlrl.tools import SqlToolset

GOLD = "SELECT name FROM customers WHERE city='NYC'"


def _q(db):
    return Question(db_id="shop", question="NYC names?", gold_sql=GOLD, db_path=db)


def test_helpers_jaccard():
    assert _schema_link_jaccard("SELECT name FROM customers", "SELECT name FROM customers") == 1.0
    assert _schema_link_jaccard("SELECT x FROM a", "SELECT y FROM b") == 0.0
    assert _bigram_jaccard("SELECT 1", "SELECT 1") == 1.0


def test_correct_sql_gets_full_seven(tiny_db):
    ts = SqlToolset(tiny_db)
    r = reward_r2(GOLD, _q(tiny_db), ts)
    assert r.ex == 1 and r.reward == 7.0
    ts.close()


def test_monotonic_correct_beats_incorrect(tiny_db):
    ts = SqlToolset(tiny_db)
    correct = reward_r2(GOLD, _q(tiny_db), ts).reward
    wrong = reward_r2("SELECT name FROM customers", _q(tiny_db), ts).reward
    assert correct > wrong
    ts.close()


def test_farmable_partial_mass_with_zero_ex(tiny_db):
    ts = SqlToolset(tiny_db)
    r = reward_r2("SELECT name FROM customers", _q(tiny_db), ts)
    assert r.ex == 0 and r.reward >= 2.5
    ts.close()


def test_invalid_sql_zero(tiny_db):
    ts = SqlToolset(tiny_db)
    r = reward_r2(None, _q(tiny_db), ts)
    assert r.reward == 0.0 and r.ex == 0
    ts.close()
