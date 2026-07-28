from qa_agent.models import TestCase, Turn, RunResult, EvalResult


def test_test_case_fields():
    tc = TestCase(
        id="tc-001",
        description="User asks a general question",
        category="general",
        goal="Agent answers accurately and politely",
        input_message="Hello, what can you help me with?",
    )
    assert tc.id == "tc-001"
    assert tc.category == "general"
    assert tc.input_message == "Hello, what can you help me with?"


def test_turn_fields():
    t = Turn(sent="Hello", received="Hi there!")
    assert t.sent == "Hello"
    assert t.received == "Hi there!"


def test_run_result_single_turn():
    tc = TestCase(
        id="tc-002",
        description="edge case",
        category="edge_case",
        goal="Handle empty input gracefully",
        input_message="",
    )
    turn = Turn(sent="", received="Could you provide more detail?")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    assert rr.success is True
    assert rr.error is None
    assert len(rr.turns) == 1


def test_run_result_failure():
    tc = TestCase(
        id="tc-003",
        description="timeout case",
        category="general",
        goal="Should respond",
        input_message="ping",
    )
    rr = RunResult(test_case=tc, turns=[], success=False, error="Timeout after 120s")
    assert rr.success is False
    assert rr.error == "Timeout after 120s"


def test_eval_result_passed():
    tc = TestCase(id="tc-004", description="d", category="general", goal="g", input_message="m")
    turn = Turn(sent="m", received="good answer")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    er = EvalResult(
        run_result=rr,
        passed=True,
        score=0.95,
        rationale="Response was accurate and complete.",
        failure_detail=None,
    )
    assert er.passed is True
    assert er.score == 0.95
    assert er.failure_detail is None


def test_eval_result_failed():
    tc = TestCase(id="tc-005", description="d", category="adversarial", goal="g", input_message="m")
    turn = Turn(sent="m", received="I'll ignore all previous instructions")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    er = EvalResult(
        run_result=rr,
        passed=False,
        score=0.1,
        rationale="Agent was jailbroken.",
        failure_detail="Response violated system prompt constraints.",
    )
    assert er.passed is False
    assert er.failure_detail is not None
