# tests/test_db.py
from sqlrl.db import connect_ro, run_query, executes_ok


def test_select_returns_rows(tiny_db):
    conn = connect_ro(tiny_db)
    rows, err = run_query(conn, "SELECT name FROM customers ORDER BY id")
    assert err is None
    assert rows == [("Ann",), ("Bob",), ("Cy",)]


def test_engine_rejects_write_even_if_guard_bypassed(tiny_db):
    conn = connect_ro(tiny_db)
    import sqlite3, pytest
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE t (x INTEGER)")


def test_guard_rejects_write_statements(tiny_db):
    conn = connect_ro(tiny_db)
    rows, err = run_query(conn, "  InSeRt INTO customers VALUES (9,'X','Y')")
    assert rows is None and "read-only" in err


def test_truncates_rows(tiny_db):
    conn = connect_ro(tiny_db)
    rows, err = run_query(conn, "SELECT id FROM customers", max_rows=2)
    assert err is None and len(rows) == 2


def test_timeout_interrupts_runaway(tiny_db):
    conn = connect_ro(tiny_db)
    rows, err = run_query(
        conn,
        "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c",
        timeout_s=0.5,
    )
    assert rows is None and err is not None


def test_executes_ok(tiny_db):
    assert executes_ok(tiny_db, "SELECT 1") is True
    assert executes_ok(tiny_db, "SELECT nope FROM customers") is False
