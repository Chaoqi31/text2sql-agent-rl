# src/sqlrl/runner.py
from concurrent.futures import ThreadPoolExecutor

from sqlrl.agent import run_agent
from sqlrl.reward import _is_select
from sqlrl.schema import RolloutResult
from sqlrl.tools import SqlToolset


def _run_one(client, model_name, q, agent_cfg, reward_fn) -> RolloutResult:
    # A transient API/db error on one question must NOT crash a full-dev eval (1534 Qs on a
    # thread pool) — score it 0 (a crashed rollout is a failure) and carry on.
    try:
        ts = SqlToolset(q.db_path)
        try:
            rollout = run_agent(client, model_name, q, ts, agent_cfg)
            rr = reward_fn(rollout.final_sql, q, ts)
            # ex_fallback: same as strict ex, except an uncommitted rollout is rescored on its
            # last run_sql query — lets one eval report both strict and salvaged EX.
            ex_fb = rr.ex
            if rollout.final_sql is None and rollout.fallback_sql:
                ex_fb = reward_fn(rollout.fallback_sql, q, ts).ex
            bd = {**rr.breakdown, "ex_fallback": ex_fb, "fallback_sql": rollout.fallback_sql}
            return RolloutResult(
                db_id=q.db_id, question=q.question, final_sql=rollout.final_sql,
                ex=rr.ex, reward=rr.reward, breakdown=bd,
                turns=rollout.turns, finished=rollout.finished)
        finally:
            ts.close()
    except Exception as e:
        return RolloutResult(
            db_id=q.db_id, question=q.question, final_sql=None,
            ex=0, reward=0.0, breakdown={"error": str(e)[:200]},
            turns=0, finished=False)


def run_questions(client, model_name, questions, *, agent_cfg, reward_fn, concurrency=1):
    """Drive run_agent -> reward per question. concurrency>1 fans the questions across a
    thread pool — each has its own toolset/sqlite connections (check_same_thread=False),
    the vLLM server batches the concurrent requests. ThreadPool.map preserves order."""
    def one(q):
        return _run_one(client, model_name, q, agent_cfg, reward_fn)
    if concurrency <= 1:
        return [one(q) for q in questions]
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, questions))


def summarize(results) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "ex_rate": sum(r.ex for r in results) / n,
        "ex_rate_fallback": sum(r.breakdown.get("ex_fallback", r.ex) for r in results) / n,
        "mean_reward": sum(r.reward for r in results) / n,
        "finished_rate": sum(1 for r in results if r.finished) / n,
        "valid_sql_rate": sum(1 for r in results if _is_select(r.final_sql)) / n,
        "avg_turns": sum(r.turns for r in results) / n,
    }
