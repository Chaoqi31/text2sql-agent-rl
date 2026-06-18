# src/sqlrl/reward.py
import re
from dataclasses import dataclass

import sqlglot
from sqlglot import exp

from sqlrl.db import executes_ok
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


# --- R2: Reasoning-SQL (arXiv 2503.23157) weights; AI-feedback (w_judge=2) omitted per spec §3. ---
R2_WEIGHTS = {"exec": 3.0, "syntax": 1.0, "schema": 1.0, "ngram": 1.0, "format": 1.0}


def _schema_items(sql: str) -> set:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return set()
    items = set()
    for t in tree.find_all(exp.Table):
        items.add(("t", (t.name or "").lower()))
    for c in tree.find_all(exp.Column):
        items.add(("c", (c.name or "").lower()))
    return items


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _schema_link_jaccard(pred_sql: str, gold_sql: str) -> float:
    return _jaccard(_schema_items(pred_sql or ""), _schema_items(gold_sql or ""))


def _bigrams(sql: str) -> set:
    toks = re.findall(r"[A-Za-z_]\w*|\S", (sql or "").lower())
    return set(zip(toks, toks[1:]))


def _bigram_jaccard(pred_sql: str, gold_sql: str) -> float:
    return _jaccard(_bigrams(pred_sql), _bigrams(gold_sql))


def reward_r2(final_sql, question, toolset, *, ex_fn=execution_match) -> RewardResult:
    valid = _is_select(final_sql)
    ex = ex_fn(final_sql, question.gold_sql, question.db_path) if valid else 0
    # ex==1 already proves the query runs; skip the redundant re-execution (and the
    # 30s-EX vs 5s-executes_ok timeout mismatch that could wrongly zero a slow-correct query)
    syntax = 1.0 if (valid and (ex == 1 or executes_ok(question.db_path, final_sql))) else 0.0
    schema = _schema_link_jaccard(final_sql, question.gold_sql) if valid else 0.0
    ngram = _bigram_jaccard(final_sql, question.gold_sql) if valid else 0.0
    fmt = 1.0 if valid else 0.0
    w = R2_WEIGHTS
    reward = (w["exec"] * ex + w["syntax"] * syntax + w["schema"] * schema
              + w["ngram"] * ngram + w["format"] * fmt)
    return RewardResult(reward, ex, {"ex": ex, "syntax": syntax, "schema": round(schema, 3),
                                     "ngram": round(ngram, 3), "format": fmt, "valid_sql": valid})


# --- append to src/sqlrl/reward.py ---
def reward_r3(final_sql, question, toolset, *, ex_fn=execution_match) -> RewardResult:
    """NAIVE process foil (spec §8.1, Pre-registration) — INTENTIONALLY gameable.
    0.2*[called describe_table] + 0.3*[emitted an executable SQL] + 1.0*[exec match].
    The first two terms can be farmed to 0.5 without solving; this is the planned
    reward-hacking exhibit, NOT a serious reward design."""
    called_describe = any(name == "describe_table" for name, _ in toolset.calls)
    executable = bool(final_sql) and executes_ok(question.db_path, final_sql)
    valid = _is_select(final_sql)
    ex = ex_fn(final_sql, question.gold_sql, question.db_path) if valid else 0
    reward = 0.2 * float(called_describe) + 0.3 * float(executable) + 1.0 * float(ex)
    return RewardResult(reward, ex, {"called_describe": called_describe,
                                     "executable": executable, "ex": ex})
