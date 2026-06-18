# src/sqlrl/train_reward.py
# Adapts the offline reward functions to TRL GRPOTrainer's reward_funcs contract:
#   reward_fn(prompts, completions, **dataset_columns) -> list[float]
# Dataset extra columns (gold_sql, db_path) arrive as keyword lists. A TRL completion is a
# str or a list of assistant message dicts — we flatten to text, extract FINAL SQL, score.
# Pure-Python (no trl/torch import) so it's CPU-unit-testable and importable by train_grpo.py.
from sqlrl.agent import extract_final_sql
from sqlrl.reward import reward_r1
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
