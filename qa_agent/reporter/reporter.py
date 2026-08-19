import json
import logging
from qa_agent.llm import call_llm
from qa_agent.models import EvalResult
from qa_agent.config import Config

logger = logging.getLogger(__name__)

_NARRATIVE_SYSTEM = """\
You are a QA analyst. Given test results for an LLM agent, write a concise narrative summary.
Cover: overall pass rate, patterns in failures (if any), and general quality assessment.
Be direct. 3-5 sentences maximum. Plain prose — no bullet points, no headers."""


def _build_narrative_prompt(eval_results: list[EvalResult], description: str | None) -> str:
    total = len(eval_results)
    passed = sum(1 for r in eval_results if r.passed)
    parts = [f"Test results: {passed}/{total} passed."]
    if description:
        parts.append(f"Agent: {description}")
    failures = [r for r in eval_results if not r.passed]
    if failures:
        parts.append("Failures:")
        for r in failures:
            parts.append(
                f"  [{r.run_result.test_case.category}] {r.run_result.test_case.description}: "
                f"{r.rationale}"
            )
    return "\n".join(parts)


def _format_results_table(eval_results: list[EvalResult]) -> str:
    header = f"{'ID':<12} {'CATEGORY':<14} {'STATUS':<8} {'SCORE':<6}"
    separator = "-" * len(header)
    rows = [header, separator]
    for r in eval_results:
        status = "PASS" if r.passed else "FAIL"
        rows.append(
            f"{r.run_result.test_case.id:<12} "
            f"{r.run_result.test_case.category:<14} "
            f"{status:<8} "
            f"{r.score:.2f}"
        )
    return "\n".join(rows)


def _format_failure_deep_dives(eval_results: list[EvalResult]) -> str:
    failures = [r for r in eval_results if not r.passed]
    if not failures:
        return ""
    lines = ["\n=== FAILURE DEEP-DIVE ==="]
    for r in failures:
        tc = r.run_result.test_case
        lines.append(f"\n[{tc.id}] {tc.description} ({tc.category})")
        lines.append(f"Goal: {tc.goal}")
        for i, turn in enumerate(r.run_result.turns, 1):
            lines.append(f"\n  Turn {i}:")
            lines.append(f"    Sent:     {turn.sent}")
            lines.append(f"    Received: {turn.received}")
            if turn.trace_url:
                lines.append(f"    Trace:    {turn.trace_url}")
            elif turn.trace_id:
                lines.append(f"    Trace ID: {turn.trace_id}")
        lines.append(f"\nRationale: {r.rationale}")
        if r.failure_detail:
            lines.append(f"Failure detail: {r.failure_detail}")
    return "\n".join(lines)


def build_run_data(
    eval_results: list[EvalResult],
    description: str | None,
    narrative: str,
    advice: str | None,
    scenario_id: str | None,
    timestamp: str | None,
    agent_context: str | None = None,
) -> dict:
    """Build the JSON-serialisable run payload used by storage and the web dashboard."""
    total = len(eval_results)
    passed = sum(1 for r in eval_results if r.passed)
    return {
        "timestamp": timestamp,
        "scenario_id": scenario_id,
        "purpose": description,
        "agent_context": agent_context,
        "summary": narrative,
        "pass_rate": passed / total if total else 0.0,
        "advice": advice,
        "results": [
            {
                "id": r.run_result.test_case.id,
                "description": r.run_result.test_case.description,
                "category": r.run_result.test_case.category,
                "goal": r.run_result.test_case.goal,
                "passed": r.passed,
                "score": r.score,
                "rationale": r.rationale,
                "failure_detail": r.failure_detail,
                "turns": [
                    {
                        "sent": t.sent,
                        "received": t.received,
                        "trace_id": t.trace_id,
                        "trace_url": t.trace_url,
                    }
                    for t in r.run_result.turns
                ],
            }
            for r in eval_results
        ],
    }


def report(
    eval_results: list[EvalResult],
    description: str | None,
    config: Config,
    scenario_id: str | None = None,
    advice: str | None = None,
    agent_context: str | None = None,
    # kept for backwards compat with existing tests
    output_path: str | None = None,
) -> tuple[int, dict]:
    """Print the report to stdout and return (exit_code, run_data)."""
    total = len(eval_results)
    passed = sum(1 for r in eval_results if r.passed)
    logger.info("Generating report: %d/%d passed", passed, total)

    narrative_prompt = _build_narrative_prompt(eval_results, description)
    narrative = call_llm(
        system=_NARRATIVE_SYSTEM,
        user=narrative_prompt,
        model=config.qa_llm.model,
        api_key=config.qa_llm_api_key,
        max_tokens=1024,
    )

    print(f"\n{'='*50}")
    print("QA AGENT REPORT")
    print(f"{'='*50}")
    print(f"\nResults: {passed}/{total} passed\n")
    print(narrative)
    print(f"\n{_format_results_table(eval_results)}")

    deep_dives = _format_failure_deep_dives(eval_results)
    if deep_dives:
        print(deep_dives)

    if advice:
        print(f"\n{'='*50}")
        print("PROMPT ADVISOR")
        print(f"{'='*50}")
        print(advice)

    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).isoformat()
    run_data = build_run_data(
        eval_results=eval_results,
        description=description,
        narrative=narrative,
        advice=advice,
        scenario_id=scenario_id,
        timestamp=timestamp,
        agent_context=agent_context,
    )

    if output_path:
        with open(output_path, "w") as f:
            json.dump(run_data, f, indent=2)
        logger.info("Report saved to %s", output_path)
        print(f"\nReport saved to: {output_path}")

    exit_code = 0 if all(r.passed for r in eval_results) else 1
    return exit_code, run_data
