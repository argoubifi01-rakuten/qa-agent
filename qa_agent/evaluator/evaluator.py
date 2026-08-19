import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from qa_agent.llm import call_llm
from qa_agent.models import RunResult, EvalResult
from qa_agent.config import Config

logger = logging.getLogger(__name__)

_MAX_PARALLEL_EVALS = 10

_SYSTEM_PROMPT = """\
You are a QA judge evaluating whether a conversation with an LLM agent met its test goal.

Return a JSON object with this exact schema:
{
  "passed": true/false,
  "score": 0.0 to 1.0,
  "rationale": "one or two sentences explaining the verdict",
  "failure_detail": "specific failure details (omit or null if passed=true)"
}

Be strict but fair. Consider whether the agent's responses were accurate, helpful, \
and consistent with its described purpose.

Return ONLY valid JSON."""


def _build_eval_prompt(
    run_result: RunResult,
    agent_context: str | None,
) -> str:
    parts = [
        f"Test case: {run_result.test_case.description}",
        f"Category: {run_result.test_case.category}",
        f"Goal: {run_result.test_case.goal}",
    ]
    if agent_context:
        parts.append(f"\nAgent context:\n{agent_context}")
    parts.append("\nConversation:")
    for turn in run_result.turns:
        parts.append(f"User: {turn.sent}")
        parts.append(f"Agent: {turn.received}")
    return "\n".join(parts)


def _evaluate_one(rr: RunResult, agent_context: str | None, config: Config) -> EvalResult:
    """Evaluate a single run result. Returns a failed EvalResult on runner failure or LLM error."""
    if not rr.success:
        logger.debug("Skipping LLM eval for failed run %s: %s", rr.test_case.id, rr.error)
        return EvalResult(
            run_result=rr,
            passed=False,
            score=0.0,
            rationale=f"Run failed: {rr.error}",
            failure_detail=rr.error,
        )

    user_msg = _build_eval_prompt(rr, agent_context)
    last_exc: Exception | None = None

    for attempt in range(1, 4):
        try:
            logger.debug("Evaluating test case %s (attempt %d)", rr.test_case.id, attempt)
            text = call_llm(
                system=_SYSTEM_PROMPT,
                user=user_msg,
                model=config.qa_llm.model,
                api_key=config.qa_llm_api_key,
                max_tokens=1024,
            )
            data = json.loads(text)
            er = EvalResult(
                run_result=rr,
                passed=bool(data.get("passed", False)),
                score=float(data.get("score", 0.0)),
                rationale=data.get("rationale", ""),
                failure_detail=data.get("failure_detail"),
            )
            logger.debug("Evaluated %s: passed=%s score=%.2f", rr.test_case.id, er.passed, er.score)
            return er
        except Exception as exc:
            last_exc = exc
            logger.warning("Evaluation attempt %d failed for %s: %s", attempt, rr.test_case.id, exc)

    logger.error("Evaluation failed for %s after 3 attempts: %s", rr.test_case.id, last_exc)
    return EvalResult(
        run_result=rr,
        passed=False,
        score=0.0,
        rationale=f"Evaluation failed after 3 attempts: {last_exc}",
        failure_detail=str(last_exc),
    )


def evaluate(
    run_results: list[RunResult],
    agent_context: str | None,
    config: Config,
) -> list[EvalResult]:
    logger.info("Evaluating %d run results in parallel (max %d workers)", len(run_results), _MAX_PARALLEL_EVALS)

    workers = min(len(run_results), _MAX_PARALLEL_EVALS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_evaluate_one, rr, agent_context, config) for rr in run_results]
        results = [f.result() for f in futures]

    passed = sum(1 for r in results if r.passed)
    logger.info("Evaluation complete: %d/%d passed", passed, len(results))
    return results
