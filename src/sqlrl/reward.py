# src/sqlrl/reward.py
from dataclasses import dataclass

from sqlrl.ex import execution_match


@dataclass
class RewardResult:
    reward: float
    ex: int                 # shadow execution-match, always logged (spec §9)
    breakdown: dict


def _is_select(sql: str | None) -> bool:
    if not sql or not sql.strip():
        return False
    low = sql.strip().lower()
    return low.startswith("select") or low.startswith("with")


def reward_r1(final_sql, question, toolset, *, ex_fn=execution_match) -> RewardResult:
    if not _is_select(final_sql):
        return RewardResult(0.0, 0, {"valid_sql": False, "ex": 0})
    ex = ex_fn(final_sql, question.gold_sql, question.db_path)
    return RewardResult(float(ex), ex, {"valid_sql": True, "ex": ex})
