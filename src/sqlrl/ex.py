# src/sqlrl/ex.py
import sqlite3
import threading

from sqlrl.db import connect_ro


def _fetch_all(conn, sql, timeout_s):
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        return conn.execute(sql).fetchall(), None
    except sqlite3.Error as e:
        return None, str(e)
    finally:
        timer.cancel()


def execution_match(pred_sql: str, gold_sql: str, db_path: str, *, timeout_s: float = 30.0) -> int:
    """BIRD official EX: 1 iff set(pred_rows) == set(gold_rows), else 0.
    Read-only execution; predicted error/timeout/empty -> 0."""
    if not pred_sql or not pred_sql.strip():
        return 0
    conn = connect_ro(db_path)
    try:
        pred_rows, perr = _fetch_all(conn, pred_sql, timeout_s)
        if perr is not None:
            return 0
        gold_rows, gerr = _fetch_all(conn, gold_sql, timeout_s)
        if gerr is not None:
            return 0
        return 1 if set(pred_rows) == set(gold_rows) else 0
    finally:
        conn.close()
