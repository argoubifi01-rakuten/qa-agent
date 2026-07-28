import pytest
from qa_agent.runner.base import Runner
from qa_agent.models import TestCase, RunResult


def test_runner_is_abstract():
    with pytest.raises(TypeError):
        Runner()


def test_runner_subclass_must_implement_run():
    class BadRunner(Runner):
        pass
    with pytest.raises(TypeError):
        BadRunner()


def test_runner_concrete_subclass_works():
    class GoodRunner(Runner):
        def run(self, test_case: TestCase) -> RunResult:
            turn_obj = __import__("qa_agent.models", fromlist=["Turn"]).Turn(
                sent=test_case.input_message, received="ok"
            )
            return RunResult(test_case=test_case, turns=[turn_obj], success=True, error=None)

    runner = GoodRunner()
    tc = TestCase(id="x", description="d", category="general", goal="g", input_message="hello")
    result = runner.run(tc)
    assert result.success is True
