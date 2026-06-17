# src/sqlrl/schema.py
from dataclasses import dataclass


@dataclass
class Question:
    db_id: str
    question: str
    gold_sql: str
    db_path: str
    evidence: str = ""
    difficulty: str = ""


@dataclass
class AgentConfig:
    max_turns: int = 8
    single_shot: bool = False          # S0: stop at first run_sql / FINAL, no self-correct
    max_tokens: int = 1024
    temperature: float = 0.0


@dataclass
class AgentRollout:
    messages: list                     # transcript dicts (for inspection)
    final_sql: str | None
    turns: int
    finished: bool
    tool_calls: list                   # list[tuple[str, dict]]


@dataclass
class RolloutResult:
    db_id: str
    question: str
    final_sql: str | None
    ex: int                            # shadow execution-match 0/1
    reward: float
    breakdown: dict
    turns: int
    finished: bool
