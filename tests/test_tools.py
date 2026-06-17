# tests/test_tools.py
from sqlrl.tools import SqlToolset


def test_list_tables(tiny_db):
    ts = SqlToolset(tiny_db)
    out = ts.list_tables()
    assert "customers" in out and "orders" in out
    ts.close()


def test_describe_table(tiny_db):
    ts = SqlToolset(tiny_db)
    out = ts.describe_table("customers")
    assert "name" in out and "city" in out and "sample" in out
    ts.close()


def test_run_sql_select(tiny_db):
    ts = SqlToolset(tiny_db)
    out = ts.run_sql("SELECT name FROM customers WHERE city='NYC' ORDER BY id")
    assert "Ann" in out and "Cy" in out
    ts.close()


def test_run_sql_write_rejected(tiny_db):
    ts = SqlToolset(tiny_db)
    assert "Error" in ts.run_sql("DELETE FROM customers")
    ts.close()


def test_dispatch_logs_calls_and_handles_errors(tiny_db):
    ts = SqlToolset(tiny_db)
    ts.dispatch("describe_table", {"table": "orders"})
    ts.dispatch("run_sql", {"query": "SELECT 1"})
    assert ts.calls == [("describe_table", {"table": "orders"}), ("run_sql", {"query": "SELECT 1"})]
    assert "unknown tool" in ts.dispatch("frobnicate", {})
    assert "missing argument" in ts.dispatch("run_sql", {})
    ts.close()


def test_reset_clears_calls(tiny_db):
    ts = SqlToolset(tiny_db)
    ts.dispatch("list_tables", {})
    ts.reset()
    assert ts.calls == []
    ts.close()


def test_describe_table_injection_rejected(tiny_db):
    ts = SqlToolset(tiny_db)
    out = ts.describe_table("customers') UNION SELECT 1,2--")
    assert "no such table" in out
    ts.close()
