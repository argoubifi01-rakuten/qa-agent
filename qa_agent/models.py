from dataclasses import dataclass, field


@dataclass
class AgentIntent:
    domain: str
    core_capabilities: list[str]
    description_prompt_gaps: list[str]
    contradictions: list[str]


@dataclass
class TestCase:
    id: str
    description: str
    category: str  # "general", "edge_case", "out_of_scope", "adversarial"
    goal: str
    input_message: str
    follow_up_messages: list[str] | None = None  # scripted turns after the first
    driver_instructions: str | None = None  # conditional guidance for the LLM driver


@dataclass
class Turn:
    sent: str
    received: str
    trace_id: str | None = None
    trace_url: str | None = None


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
