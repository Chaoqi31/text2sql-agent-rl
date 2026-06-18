# src/sqlrl/tools.py
from sqlrl.db import connect_ro, run_query


class SqlToolset:
    def __init__(self, db_path: str, *, max_rows: int = 50, max_cell: int = 200, timeout_s: float = 5.0):
        self.db_path = db_path
        self.max_rows = max_rows
        self.max_cell = max_cell
        self.timeout_s = timeout_s
        self.conn = connect_ro(db_path)
        self.reset()

    def reset(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def close(self) -> None:
        self.conn.close()

    def _table_names(self) -> list[str]:
        rows, err = run_query(
            self.conn, "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name",
            timeout_s=self.timeout_s, max_rows=1000, max_cell=self.max_cell)
        return [] if err else [r[0] for r in rows]

    def list_tables(self) -> str:
        names = self._table_names()
        return "Tables: " + ", ".join(names) if names else "No tables."

    def describe_table(self, table: str, sample_rows: int = 3) -> str:
        if table not in self._table_names():
            return f"Error: no such table: {table}"
        cols, err = run_query(
            self.conn, f"SELECT name, type FROM pragma_table_info('{table}')",
            timeout_s=self.timeout_s, max_rows=500, max_cell=self.max_cell)
        if err:
            return f"Error: {err}"
        if not cols:
            return f"Error: no such table: {table}"
        header = ", ".join(f"{c[0]} {c[1]}" for c in cols)
        sample, serr = run_query(
            self.conn, f'SELECT * FROM "{table}" LIMIT {int(sample_rows)}',
            timeout_s=self.timeout_s, max_rows=sample_rows, max_cell=self.max_cell)
        sample_str = "(unavailable)" if serr else "\n".join(str(r) for r in sample)
        return f"Table {table}:\ncolumns: {header}\nsample:\n{sample_str}"

    def run_sql(self, query: str) -> str:
        # fetch one extra row so we can tell a full result from a truncated one
        rows, err = run_query(self.conn, query, timeout_s=self.timeout_s,
                              max_rows=self.max_rows + 1, max_cell=self.max_cell)
        if err:
            return f"Error: {err}"
        if not rows:
            return "(0 rows)"
        truncated = len(rows) > self.max_rows
        rows = rows[:self.max_rows]
        body = "\n".join(str(r) for r in rows)
        note = f"\n(showing first {self.max_rows})" if truncated else ""
        return f"{len(rows)} row(s):\n{body}{note}"

    def dispatch(self, name: str, args: dict) -> str:
        self.calls.append((name, dict(args)))
        try:
            if name == "list_tables":
                return self.list_tables()
            if name == "describe_table":
                return self.describe_table(args["table"], args.get("sample_rows", 3))
            if name == "run_sql":
                return self.run_sql(args["query"])
            return f"Error: unknown tool {name!r}"
        except KeyError as e:
            return f"Error: missing argument {e}"

    @property
    def tool_specs(self) -> list[dict]:
        def fn(name, desc, props, required):
            return {"type": "function", "function": {
                "name": name, "description": desc,
                "parameters": {"type": "object", "properties": props, "required": required}}}
        return [
            fn("list_tables", "List all table names in the database.", {}, []),
            fn("describe_table", "Show columns, types, and a few sample rows for one table.",
               {"table": {"type": "string"}}, ["table"]),
            fn("run_sql", "Run a read-only SELECT and return result rows (truncated).",
               {"query": {"type": "string"}}, ["query"]),
        ]
