# tests/test_prepare_sft.py
from scripts.prepare_sft import bird_to_sft, db_schema_text, synsql_to_sft
from sqlrl.prompts import SYSTEM_PROMPT
from sqlrl.schema import Question


def test_db_schema_text_has_create_tables(tiny_db):
    s = db_schema_text(tiny_db)
    assert "CREATE TABLE customers" in s and "CREATE TABLE orders" in s


def test_bird_to_sft_shape(tiny_db):
    q = Question(db_id="shop", question="names in NYC?",
                 gold_sql="SELECT name FROM customers WHERE city='NYC'", db_path=tiny_db)
    ex = bird_to_sft(q)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == SYSTEM_PROMPT
    assert "CREATE TABLE customers" in ex["messages"][1]["content"]
    assert ex["messages"][2]["content"] == "FINAL SQL: SELECT name FROM customers WHERE city='NYC'"


def test_synsql_to_sft_joins_tables_and_builds_target():
    rec = {"db_id": "shop", "question": "q?", "external_knowledge": "ev",
           "cot": "step 1", "sql": "SELECT a FROM t"}
    tables = {"shop": "CREATE TABLE t(a INT)"}
    ex = synsql_to_sft(rec, tables)
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][2]["content"] == "step 1\nFINAL SQL: SELECT a FROM t"
    assert "CREATE TABLE t(a INT)" in ex["messages"][1]["content"]
    assert "Evidence: ev" in ex["messages"][1]["content"]
