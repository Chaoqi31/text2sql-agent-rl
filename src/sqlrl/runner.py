# src/sqlrl/runner.py
from sqlrl.agent import run_agent
from sqlrl.schema import RolloutResult
from sqlrl.tools import SqlToolset


def run_questions(client, model_name, questions, *, agent_cfg, reward_fn, concurrency=1):
    """Sync driver. `concurrency` reserved for the plan-2 async vLLM client."""
    results: list[RolloutResult] = []
    for q in questions:
        ts = SqlToolset(q.db_path)
        try:
            rollout = run_agent(client, model_name, q, ts, agent_cfg)
            rr = reward_fn(rollout.final_sql, q, ts)
            results.append(RolloutResult(
                db_id=q.db_id, question=q.question, final_sql=rollout.final_sql,
                ex=rr.ex, reward=rr.reward, breakdown=rr.breakdown,
                turns=rollout.turns, finished=rollout.finished))
        finally:
            ts.close()
    return results


def summarize(results) -> dict:
    n = len(results) or 1
    return {
        "n": len(results),
        "ex_rate": sum(r.ex for r in results) / n,
        "mean_reward": sum(r.reward for r in results) / n,
        "finished_rate": sum(1 for r in results if r.finished) / n,
        "valid_sql_rate": sum(1 for r in results if r.final_sql) / n,
        "avg_turns": sum(r.turns for r in results) / n,
    }
