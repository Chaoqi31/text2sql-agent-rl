# src/sqlrl/train_reward.py
# Adapts the offline reward functions to TRL GRPOTrainer's reward_funcs contract:
#   reward_fn(prompts, completions, **dataset_columns) -> list[float]
# Dataset extra columns (gold_sql, db_path) arrive as keyword lists. A TRL completion is a
# str or a list of assistant message dicts — we flatten to text, extract FINAL SQL, score.
# Pure-Python (no trl/torch import) so it's CPU-unit-testable and importable by train_grpo.py.
from sqlrl.agent import extract_final_sql
from sqlrl.reward import reward_r1, reward_r2, reward_r3
from sqlrl.schema import Question


def _completion_text(completion) -> str:
    if isinstance(completion, str):
        return completion
    parts: list[str] = []
    for msg in completion:
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str):
                parts.append(content)
    return "\n".join(parts)


def _completion_tool_names(completion) -> list[str]:
    """Tool-call names from a TRL completion's assistant messages (for R3's describe_table signal)."""
    names: list[str] = []
    if isinstance(completion, str):
        return names
    for msg in completion:
        if not isinstance(msg, dict):
            continue
        for tc in (msg.get("tool_calls") or []):
            fn = tc.get("function") if isinstance(tc, dict) else None
            n = fn.get("name") if isinstance(fn, dict) else None
            if n:
                names.append(n)
    return names


class _StubToolset:
    """Minimal stand-in exposing `.calls` so reward_r3's logic can be reused at TRL reward time
    (the real toolset isn't passed to TRL reward functions)."""
    def __init__(self, calls):
        self.calls = calls


def make_r1_reward(*, ex_fn=None):
    """R1 (execution-only) reward for GRPO. reward = BIRD EX of the emitted FINAL SQL."""

    def r1_reward(prompts=None, completions=None, gold_sql=None, db_path=None, **kwargs):
        rewards: list[float] = []
        for comp, gold, dbp in zip(completions, gold_sql, db_path):
            final_sql = extract_final_sql(_completion_text(comp))
            q = Question(db_id="", question="", gold_sql=gold, db_path=dbp)
            r = reward_r1(final_sql, q, None, ex_fn=ex_fn) if ex_fn else reward_r1(final_sql, q, None)
            rewards.append(r.reward)
        return rewards

    r1_reward.__name__ = "r1_reward"
    return r1_reward


def make_r2_reward(*, ex_fn=None):
    """R2 (faithful partial) reward for GRPO. R2 reads no toolset -> pure (final_sql, gold, db)."""

    def r2_reward(prompts=None, completions=None, gold_sql=None, db_path=None, **kwargs):
        rewards: list[float] = []
        for comp, gold, dbp in zip(completions, gold_sql, db_path):
            final_sql = extract_final_sql(_completion_text(comp))
            q = Question(db_id="", question="", gold_sql=gold, db_path=dbp)
            r = reward_r2(final_sql, q, None, ex_fn=ex_fn) if ex_fn else reward_r2(final_sql, q, None)
            rewards.append(r.reward)
        return rewards

    r2_reward.__name__ = "r2_reward"
    return r2_reward


def make_r3_reward(*, ex_fn=None):
    """R3 (naive process foil) reward for GRPO. The 'called describe_table' signal is parsed
    from the completion's tool calls (toolset.calls isn't available at TRL reward time)."""

    def r3_reward(prompts=None, completions=None, gold_sql=None, db_path=None, **kwargs):
        rewards: list[float] = []
        for comp, gold, dbp in zip(completions, gold_sql, db_path):
            final_sql = extract_final_sql(_completion_text(comp))
            called = ("describe_table" in _completion_tool_names(comp)
                      or "describe_table" in _completion_text(comp))
            stub = _StubToolset([("describe_table", {})] if called else [])
            q = Question(db_id="", question="", gold_sql=gold, db_path=dbp)
            r = reward_r3(final_sql, q, stub, ex_fn=ex_fn) if ex_fn else reward_r3(final_sql, q, stub)
            rewards.append(r.reward)
        return rewards

    r3_reward.__name__ = "r3_reward"
    return r3_reward
