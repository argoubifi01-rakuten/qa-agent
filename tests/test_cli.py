import sys
import pytest
from unittest.mock import MagicMock, patch
from qa_agent.cli import main

_MOCK_RUN_DATA = {"summary": "ok", "pass_rate": 1.0, "results": [], "scenario_id": "my-scenario"}
_AGENT_CONTEXT = "# Agent: Test Bot\n## Purpose\nHelp customers."

_REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "EVAL_MOCK_SECRET": "mock-s",
    "MONGODB_URI": "mongodb://localhost/testdb",
}

_MINIMAL_YAML = """\
qa_llm:
  provider: openai
  model: gpt-4o
target:
  websocket_url: wss://x/ws
  auth_url: https://x/auth
  thread_creation_url: https://x/threads
  scenario_id: test-scenario
  wss_response_timeout: 30
test_generation:
  num_test_cases: 2
  max_turns: 1
"""

_NO_BROWSER = "--no-browser"

# Common patches shared by most run tests
_BASE_RUN_PATCHES = [
    ("qa_agent.cli.load_context", {"return_value": _AGENT_CONTEXT}),
    ("qa_agent.cli.fetch_system_prompt", {"return_value": "You are a helpful bot."}),
    ("qa_agent.cli.generate_test_cases", {"return_value": [MagicMock()]}),
    ("qa_agent.cli.evaluate", {}),
    ("qa_agent.cli.advise", {"return_value": "Good prompt."}),
    ("qa_agent.cli.report", {"return_value": (0, _MOCK_RUN_DATA)}),
    ("qa_agent.cli.save_run", {"return_value": "runs/fake.json"}),
    ("qa_agent.cli.load_scenario_runs", {"return_value": []}),
    ("qa_agent.cli.update_context", {"return_value": _AGENT_CONTEXT}),
]


def _make_eval_results(passed=True):
    from qa_agent.models import TestCase, Turn, RunResult, EvalResult
    tc = TestCase(id="tc-001", description="d", category="general", goal="g", input_message="hi")
    turn = Turn(sent="hi", received="hello")
    rr = RunResult(test_case=tc, turns=[turn], success=True, error=None)
    return [EvalResult(run_result=rr, passed=passed, score=0.9 if passed else 0.1,
                       rationale="r", failure_detail=None if passed else "detail")]


def _run_test(argv, extra_patches=None, eval_passed=True):
    """Helper: run main() with mocked argv, env, and pipeline. Returns SystemExit code."""
    eval_results = _make_eval_results(passed=eval_passed)
    patches = dict(_BASE_RUN_PATCHES)
    patches["qa_agent.cli.evaluate"]["return_value"] = eval_results
    if extra_patches:
        patches.update(extra_patches)

    ctx = [
        patch("sys.argv", argv),
        patch.dict("os.environ", _REQUIRED_ENV),
    ]
    for target, kwargs in patches.items():
        ctx.append(patch(target, **kwargs))

    # WebSocketRunner always needs special handling
    mock_runner = MagicMock()
    mock_runner.run.return_value = MagicMock()
    ctx.append(patch("qa_agent.cli.WebSocketRunner", return_value=mock_runner))

    from contextlib import ExitStack
    with ExitStack() as stack:
        for p in ctx:
            stack.enter_context(p)
        with pytest.raises(SystemExit) as exc_info:
            main()
    return exc_info.value.code


def test_cli_requires_scenario_id_when_no_browser(capsys):
    with patch("sys.argv", [
        "qa-agent", "run",
        "--scenario-id", "sc",
        "--purpose", "p",
        _NO_BROWSER,
        "--config", "nonexistent.yaml",
    ]), patch.dict("os.environ", _REQUIRED_ENV):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


def test_cli_all_pass_exits_0(tmp_path):
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    code = _run_test(
        ["qa-agent", "run", "--scenario-id", "my-scenario", "--purpose", "Help customers",
         "--config", str(config_file), _NO_BROWSER],
        eval_passed=True,
    )
    assert code == 0


def test_cli_any_fail_exits_1(tmp_path):
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    code = _run_test(
        ["qa-agent", "run", "--scenario-id", "my-scenario", "--purpose", "A bot",
         "--config", str(config_file), _NO_BROWSER],
        extra_patches={"qa_agent.cli.report": {"return_value": (1, _MOCK_RUN_DATA)}},
        eval_passed=False,
    )
    assert code == 1


def test_cli_scenario_id_overrides_config(tmp_path):
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    captured_scenario_ids = []

    def capture_fetch(scenario_id, mongodb_uri):
        captured_scenario_ids.append(scenario_id)
        return "You are a bot."

    eval_results = _make_eval_results(passed=True)
    with patch("sys.argv", [
        "qa-agent", "run",
        "--scenario-id", "cli-scenario-override",
        "--purpose", "Some purpose",
        "--config", str(config_file),
        _NO_BROWSER,
    ]), \
    patch.dict("os.environ", _REQUIRED_ENV), \
    patch("qa_agent.cli.load_context", return_value=_AGENT_CONTEXT), \
    patch("qa_agent.cli.fetch_system_prompt", side_effect=capture_fetch), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.advise", return_value="advice"), \
    patch("qa_agent.cli.report", return_value=(0, _MOCK_RUN_DATA)), \
    patch("qa_agent.cli.save_run", return_value="runs/fake.json"), \
    patch("qa_agent.cli.load_scenario_runs", return_value=[]), \
    patch("qa_agent.cli.update_context", return_value=_AGENT_CONTEXT):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        try:
            main()
        except SystemExit:
            pass
    assert captured_scenario_ids[0] == "cli-scenario-override"


def test_cli_no_advisor_when_prompt_missing(tmp_path):
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    eval_results = _make_eval_results()

    with patch("sys.argv", [
        "qa-agent", "run",
        "--scenario-id", "my-scenario",
        "--purpose", "Some purpose",
        "--config", str(config_file),
        _NO_BROWSER,
    ]), \
    patch.dict("os.environ", _REQUIRED_ENV), \
    patch("qa_agent.cli.load_context", return_value=_AGENT_CONTEXT), \
    patch("qa_agent.cli.fetch_system_prompt", return_value=None), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.advise") as mock_advise, \
    patch("qa_agent.cli.report", return_value=(0, _MOCK_RUN_DATA)), \
    patch("qa_agent.cli.save_run", return_value="runs/fake.json"), \
    patch("qa_agent.cli.load_scenario_runs", return_value=[]), \
    patch("qa_agent.cli.update_context", return_value=_AGENT_CONTEXT):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        try:
            main()
        except SystemExit:
            pass
    mock_advise.assert_not_called()


def test_cli_onboarding_triggered_when_no_context(tmp_path):
    """When load_context returns None, run_onboarding is called automatically."""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    eval_results = _make_eval_results(passed=True)

    with patch("sys.argv", [
        "qa-agent", "run",
        "--scenario-id", "new-scenario",
        "--purpose", "New bot",
        "--config", str(config_file),
        _NO_BROWSER,
    ]), \
    patch.dict("os.environ", _REQUIRED_ENV), \
    patch("qa_agent.cli.load_context", return_value=None), \
    patch("qa_agent.cli.fetch_agent_data", return_value={"name": "Bot", "prompt": "", "tools": []}), \
    patch("qa_agent.cli.run_onboarding", return_value=(_AGENT_CONTEXT, "runs/new-scenario/agent-context.md")) as mock_onboard, \
    patch("qa_agent.cli.fetch_system_prompt", return_value="prompt"), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.advise", return_value="advice"), \
    patch("qa_agent.cli.report", return_value=(0, _MOCK_RUN_DATA)), \
    patch("qa_agent.cli.save_run", return_value="runs/fake.json"), \
    patch("qa_agent.cli.load_scenario_runs", return_value=[]), \
    patch("qa_agent.cli.update_context", return_value=_AGENT_CONTEXT):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        try:
            main()
        except SystemExit:
            pass
    mock_onboard.assert_called_once()


def test_cli_interactive_opens_browser_and_serve(tmp_path):
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    eval_results = _make_eval_results(passed=True)

    with patch("sys.argv", [
        "qa-agent", "run",
        "--scenario-id", "my-scenario",
        "--purpose", "Help customers",
        "--config", str(config_file),
    ]), \
    patch.dict("os.environ", _REQUIRED_ENV), \
    patch("qa_agent.cli.load_context", return_value=_AGENT_CONTEXT), \
    patch("qa_agent.cli.fetch_system_prompt", return_value="prompt"), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.advise", return_value="advice"), \
    patch("qa_agent.cli.report", return_value=(0, _MOCK_RUN_DATA)), \
    patch("qa_agent.cli.save_run", return_value="runs/fake.json"), \
    patch("qa_agent.cli.load_scenario_runs", return_value=[]), \
    patch("qa_agent.cli.update_context", return_value=_AGENT_CONTEXT), \
    patch("qa_agent.cli.serve") as mock_serve, \
    patch("qa_agent.cli.webbrowser.open") as mock_browser, \
    patch("qa_agent.cli.time.sleep"):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        mock_serve.return_value = None
        fake_thread = MagicMock()
        fake_thread.is_alive.return_value = False
        with patch("qa_agent.cli.threading.Thread", return_value=fake_thread):
            with pytest.raises(SystemExit) as exc_info:
                main()

    mock_browser.assert_called_once()
    assert exc_info.value.code == 0


def test_cli_iterate_subcommand(tmp_path):
    """iterate subcommand runs pipeline non-interactively and exits."""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    code = _run_test(
        ["qa-agent", "iterate",
         "--scenario-id", "my-scenario",
         "--purpose", "Help customers",
         "--config", str(config_file)],
        eval_passed=True,
    )
    assert code == 0


def test_cli_iterate_loads_prior_runs(tmp_path):
    """iterate passes iteration history to advise()."""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    prior = [{"iteration": 1, "pass_rate": 0.5, "results": [], "advice": ""}]
    captured = {}

    def capture_advise(**kwargs):
        captured.update(kwargs)
        return "advice"

    eval_results = _make_eval_results(passed=True)
    with patch("sys.argv", [
        "qa-agent", "iterate",
        "--scenario-id", "my-scenario",
        "--purpose", "Help customers",
        "--config", str(config_file),
    ]), \
    patch.dict("os.environ", _REQUIRED_ENV), \
    patch("qa_agent.cli.load_context", return_value=_AGENT_CONTEXT), \
    patch("qa_agent.cli.fetch_system_prompt", return_value="prompt"), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.advise", side_effect=lambda **kw: capture_advise(**kw) or "advice"), \
    patch("qa_agent.cli.report", return_value=(0, _MOCK_RUN_DATA)), \
    patch("qa_agent.cli.save_run", return_value="runs/fake.json"), \
    patch("qa_agent.cli.load_scenario_runs", return_value=prior), \
    patch("qa_agent.cli.update_context", return_value=_AGENT_CONTEXT):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        try:
            main()
        except SystemExit:
            pass

    assert captured.get("iteration_history") == prior


def test_cli_history_subcommand_no_runs(tmp_path, capsys):
    runs_dir = str(tmp_path / "runs")
    with patch("sys.argv", [
        "qa-agent", "history",
        "--scenario-id", "my-scenario",
        "--runs-dir", runs_dir,
    ]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "No runs found" in captured.out


def test_cli_history_subcommand_with_runs(tmp_path, capsys):
    runs_dir = tmp_path / "runs"
    scenario_dir = runs_dir / "my-scenario"
    scenario_dir.mkdir(parents=True)
    import json
    run1 = {
        "iteration": 1, "timestamp": "2026-08-14T10:00:00+00:00",
        "pass_rate": 0.5, "results": [{"passed": True}, {"passed": False}],
    }
    (scenario_dir / "run-001-20260814-100000.json").write_text(json.dumps(run1))

    with patch("sys.argv", [
        "qa-agent", "history",
        "--scenario-id", "my-scenario",
        "--runs-dir", str(runs_dir),
    ]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "my-scenario" in captured.out
    assert "1/2" in captured.out


def test_cli_run_data_includes_iteration(tmp_path):
    """The iteration field is written into run_data before save_run is called."""
    config_file = tmp_path / "qa_agent.yaml"
    config_file.write_text(_MINIMAL_YAML)
    saved_data = {}

    def capture_save(run_data, runs_dir="runs"):
        saved_data.update(run_data)
        return "runs/fake.json"

    prior = [{"iteration": 1, "pass_rate": 0.4, "results": []}]
    eval_results = _make_eval_results(passed=True)

    with patch("sys.argv", [
        "qa-agent", "iterate",
        "--scenario-id", "my-scenario",
        "--purpose", "Help",
        "--config", str(config_file),
    ]), \
    patch.dict("os.environ", _REQUIRED_ENV), \
    patch("qa_agent.cli.load_context", return_value=_AGENT_CONTEXT), \
    patch("qa_agent.cli.fetch_system_prompt", return_value="prompt"), \
    patch("qa_agent.cli.generate_test_cases", return_value=[MagicMock()]), \
    patch("qa_agent.cli.WebSocketRunner") as mock_runner_cls, \
    patch("qa_agent.cli.evaluate", return_value=eval_results), \
    patch("qa_agent.cli.advise", return_value="advice"), \
    patch("qa_agent.cli.report", return_value=(0, {"summary": "ok", "pass_rate": 1.0,
                                                    "results": [], "scenario_id": "my-scenario"})), \
    patch("qa_agent.cli.save_run", side_effect=capture_save), \
    patch("qa_agent.cli.load_scenario_runs", return_value=prior), \
    patch("qa_agent.cli.update_context", return_value=_AGENT_CONTEXT):
        mock_runner = MagicMock()
        mock_runner.run.return_value = MagicMock()
        mock_runner_cls.return_value = mock_runner
        try:
            main()
        except SystemExit:
            pass

    assert saved_data.get("iteration") == 2
