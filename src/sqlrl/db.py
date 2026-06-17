# src/sqlrl/db.py
import sqlite3
import threading

# Read-only is engine-enforced: mode=ro URI + PRAGMA query_only + an authorizer that denies ATTACH.
# _WRITE_TOKENS below is a secondary guard for clearer error messages on obvious writes.
_WRITE_TOKENS = ("insert", "update", "delete", "drop", "alter", "create", "replace",
                 "attach", "detach", "pragma", "vacuum", "reindex", "truncate", "begin",
                 "commit")


def _deny_attach(action, arg1, arg2, db_name, trigger):
    return sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_ATTACH else sqlite3.SQLITE_OK


def connect_ro(db_path: str) -> sqlite3.Connection:
    # check_same_thread=False so the timeout Timer thread may call conn.interrupt()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_deny_attach)
    return conn


def _looks_like_write(sql: str) -> bool:
    low = sql.lstrip().lower()
    return any(low.startswith(t) for t in _WRITE_TOKENS)


def _cap_cell(value, max_cell: int) -> str:
    s = "NULL" if value is None else str(value)
    return s if len(s) <= max_cell else s[:max_cell] + "…"


def run_query(conn, sql, *, timeout_s: float = 5.0, max_rows: int = 50, max_cell: int = 200):
    """Read-only execute for TOOL DISPLAY. Returns (rows_as_str_tuples | None, error | None).
    Cells are stringified + capped — do NOT use for EX (use ex.execution_match)."""
    if _looks_like_write(sql):
        return None, "read-only: write statements are rejected"
    timer = threading.Timer(timeout_s, conn.interrupt)
    timer.start()
    try:
        cur = conn.execute(sql)
        raw = cur.fetchmany(max_rows)
    except sqlite3.Error as e:
        return None, f"SQL error: {e}"
    finally:
        timer.cancel()
    rows = [tuple(_cap_cell(c, max_cell) for c in r) for r in raw]
    return rows, None


def executes_ok(db_path: str, sql: str, *, timeout_s: float = 5.0) -> bool:
    """True if sql runs read-only without error (used by R2 syntax/executable, R3)."""
    if not sql or not sql.strip():
        return False
    conn = connect_ro(db_path)
    try:
        _, err = run_query(conn, sql, timeout_s=timeout_s, max_rows=1)
        return err is None
    finally:
        conn.close()
