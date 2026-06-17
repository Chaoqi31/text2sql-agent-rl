# tests/test_ex.py
from sqlrl.ex import execution_match


def test_match_is_order_insensitive(tiny_db):
    pred = "SELECT name FROM customers ORDER BY id"
    gold = "SELECT name FROM customers ORDER BY name DESC"
    assert execution_match(pred, gold, tiny_db) == 1


def test_mismatch_returns_zero(tiny_db):
    pred = "SELECT name FROM customers WHERE city='LA'"
    gold = "SELECT name FROM customers"
    assert execution_match(pred, gold, tiny_db) == 0


def test_pred_error_returns_zero(tiny_db):
    assert execution_match("SELECT nope FROM customers", "SELECT 1", tiny_db) == 0


def test_empty_pred_returns_zero(tiny_db):
    assert execution_match("", "SELECT 1", tiny_db) == 0


def test_write_pred_blocked_returns_zero(tiny_db):
    assert execution_match("CREATE TABLE t(x)", "SELECT 1", tiny_db) == 0
