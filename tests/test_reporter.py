import json
import pytest
from unittest.mock import patch
from qa_agent.models import TestCase, Turn, RunResult, EvalResult
from qa_agent.reporter.reporter import report
from qa_agent.config import Config, QALLMConfig, TargetConfig, TestGenerationConfig


@pytest.fixture
def config():
    return Config(
        qa_llm=QALLMConfig(provider="openai", model="gpt-4o"),
        target=TargetConfig(
            websocket_url="wss://x/ws",
            auth_url="https://x/auth",
            thread_creation_url="https://x/threads",
            scenario_id="test-scenario",
            wss_response_timeout=30,
        ),
        test_generation=TestGenerationConfig(num_test_cases=2, max_turns=1),
        qa_llm_api_key="test-key",
        eval_mock_secret="mock-secret",
        mongodb_uri="mongodb://localhost/testdb",
    )


def _make_eval_result(tc_id, category, passed, score, failure_detail=None):
    tc = TestCase(
        id=tc_id, description=f"Test {tc_id}", category=category,
        goal="goal", input_message="hello",
    )
    turn = Turn(sent="hello", received="response")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    return EvalResult(
        run_result=rr, passed=passed, score=score,
        rationale="Some rationale.", failure_detail=failure_detail,
    )


def test_report_all_pass_returns_exit_0(config, capsys):
    results = [
        _make_eval_result("tc-001", "general", passed=True, score=0.9),
        _make_eval_result("tc-002", "edge_case", passed=True, score=0.8),
    ]
    with patch("qa_agent.reporter.reporter.call_llm",
               return_value="Agent performed well across all categories."):
        exit_code, _ = report(results, description="customer bot", config=config)
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "tc-001" in captured.out
    assert "tc-002" in captured.out
    assert "PASS" in captured.out


def test_report_any_fail_returns_exit_1(config, capsys):
    results = [
        _make_eval_result("tc-001", "general", passed=True, score=0.9),
        _make_eval_result(
            "tc-002", "adversarial", passed=False, score=0.1,
            failure_detail="Agent was jailbroken.",
        ),
    ]
    with patch("qa_agent.reporter.reporter.call_llm", return_value="One test failed."):
        exit_code, _ = report(results, description="bot", config=config)
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "FAIL" in captured.out
    assert "Agent was jailbroken." in captured.out


def test_report_passing_cases_not_deep_dived(config, capsys):
    results = [_make_eval_result("tc-001", "general", passed=True, score=0.9)]
    with patch("qa_agent.reporter.reporter.call_llm", return_value="All good."):
        report(results, description="bot", config=config)
    captured = capsys.readouterr()
    assert "FAILURE DEEP-DIVE" not in captured.out


def test_report_writes_json_output(config, tmp_path, capsys):
    output_file = str(tmp_path / "report.json")
    results = [
        _make_eval_result("tc-001", "general", passed=True, score=0.95),
        _make_eval_result(
            "tc-002", "edge_case", passed=False, score=0.2,
            failure_detail="Wrong answer.",
        ),
    ]
    with patch("qa_agent.reporter.reporter.call_llm", return_value="Mixed results."):
        report(results, description="bot", config=config, output_path=output_file)
    with open(output_file) as f:
        data = json.load(f)
    assert "summary" in data
    assert "results" in data
    assert len(data["results"]) == 2
    assert data["results"][0]["id"] == "tc-001"
    assert data["results"][1]["passed"] is False


def test_report_shows_trace_url_in_failure_deep_dive(config, capsys):
    tc = TestCase(id="tc-003", description="Test trace", category="general", goal="goal", input_message="hi")
    turn = Turn(
        sent="hi", received="response",
        trace_id="abc123", trace_url="https://traces.example.com/abc123",
    )
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    result = EvalResult(
        run_result=rr, passed=False, score=0.1,
        rationale="Failed.", failure_detail="Something went wrong.",
    )
    with patch("qa_agent.reporter.reporter.call_llm", return_value="One failure."):
        report([result], description="bot", config=config)
    captured = capsys.readouterr()
    assert "https://traces.example.com/abc123" in captured.out


def test_report_json_includes_trace_fields(config, tmp_path):
    output_file = str(tmp_path / "report.json")
    tc = TestCase(id="tc-004", description="Trace test", category="general", goal="goal", input_message="hi")
    turn = Turn(
        sent="hi", received="response",
        trace_id="xyz789", trace_url="https://traces.example.com/xyz789",
    )
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    result = EvalResult(run_result=rr, passed=True, score=0.9, rationale="Good.", failure_detail=None)
    with patch("qa_agent.reporter.reporter.call_llm", return_value="All pass."):
        report([result], description="bot", config=config, output_path=output_file)
    with open(output_file) as f:
        data = json.load(f)
    turn_data = data["results"][0]["turns"][0]
    assert turn_data["trace_id"] == "xyz789"
    assert turn_data["trace_url"] == "https://traces.example.com/xyz789"


def test_report_shows_advice_section(config, capsys):
    results = [_make_eval_result("tc-001", "general", passed=False, score=0.3)]
    with patch("qa_agent.reporter.reporter.call_llm", return_value="Needs improvement."):
        report(
            results, description="bot", config=config,
            advice="ASSESSMENT\nThe prompt is vague.\n\nPROPOSED PROMPT\nBe more specific.",
        )
    captured = capsys.readouterr()
    assert "PROMPT ADVISOR" in captured.out
    assert "The prompt is vague." in captured.out


def test_report_advice_in_json(config, tmp_path):
    output_file = str(tmp_path / "report.json")
    results = [_make_eval_result("tc-001", "general", passed=True, score=0.9)]
    with patch("qa_agent.reporter.reporter.call_llm", return_value="Good."):
        report(
            results, description="bot", config=config, output_path=output_file,
            advice="Some advice.",
        )
    with open(output_file) as f:
        data = json.load(f)
    assert data["advice"] == "Some advice."


def test_report_returns_run_data_with_metadata(config):
    results = [_make_eval_result("tc-001", "general", passed=True, score=0.9)]
    with patch("qa_agent.reporter.reporter.call_llm", return_value="Good."):
        exit_code, run_data = report(
            results, description="my bot", config=config,
            scenario_id="sc-001",
        )
    assert exit_code == 0
    assert run_data["scenario_id"] == "sc-001"
    assert run_data["purpose"] == "my bot"
    assert "timestamp" in run_data
    assert run_data["results"][0]["goal"] == "goal"
