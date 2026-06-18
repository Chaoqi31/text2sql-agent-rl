# src/sqlrl/train_env.py
# TRL GRPOTrainer environment_factory for training rollouts. Each public method below
# becomes a tool (docstring + type hints -> the JSON schema the model sees); TRL drives
# generate -> call tool -> feed result -> repeat and masks tool-result tokens from the loss.
#
# Unlike prior-project's shared corpus, every BIRD question has its OWN database, so the env binds
# its db_path per rollout in reset(). Whether GRPOTrainer passes dataset row columns to
# reset(**kwargs) is the Phase -1 unknown (plan-2): if it does, db_path arrives here; if not,
# the binding moves to a custom rollout. Tool NAMES match SqlToolset.tool_specs (fairness).
from sqlrl.tools import SqlToolset


def make_sql_env_factory(*, max_rows: int = 50, max_cell: int = 200, timeout_s: float = 5.0):
    """Return a zero-arg factory building a fresh per-rollout SQL tool environment."""

    class SqlEnv:
        def __init__(self) -> None:
            self._ts: SqlToolset | None = None

        def reset(self, db_path: str | None = None, **kwargs) -> None:
            if self._ts is not None:
                self._ts.close()
            self._ts = (
                SqlToolset(db_path, max_rows=max_rows, max_cell=max_cell, timeout_s=timeout_s)
                if db_path else None
            )
            return None

        @property
        def calls(self):
            return [] if self._ts is None else self._ts.calls

        def list_tables(self) -> str:
            """List all table names in the database."""
            return self._ts.list_tables()

        def describe_table(self, table: str) -> str:
            """Show columns, types, and a few sample rows for one table.

            Args:
                table: Exact table name (as returned by list_tables).
            """
            return self._ts.describe_table(table)

        def run_sql(self, query: str) -> str:
            """Run a read-only SELECT and return result rows (truncated).

            Args:
                query: A single read-only SELECT (or WITH) query.
            """
            return self._ts.run_sql(query)

    return SqlEnv
