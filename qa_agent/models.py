from dataclasses import dataclass


@dataclass
class TestCase:
    id: str
    description: str
    category: str  # "general", "edge_case", "out_of_scope", "adversarial"
    goal: str
    input_message: str


@dataclass
class Turn:
    sent: str
    received: str


@dataclass
class RunResult:
    test_case: TestCase
    turns: list[Turn]
    success: bool
    error: str | None


@dataclass
class EvalResult:
    run_result: RunResult
    passed: bool
    score: float  # 0.0–1.0
    rationale: str
    failure_detail: str | None  # populated only when passed=False
