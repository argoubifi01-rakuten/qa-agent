import json
import anthropic
from qa_agent.models import EvalResult
from qa_agent.config import Config

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
    lines = ["\n=== FAILURE DEEP-DIVES ==="]
    for r in failures:
        tc = r.run_result.test_case
        lines.append(f"\n[{tc.id}] {tc.description} ({tc.category})")
        lines.append(f"Goal: {tc.goal}")
        for i, turn in enumerate(r.run_result.turns, 1):
            lines.append(f"\n  Turn {i}:")
            lines.append(f"    Sent:     {turn.sent}")
            lines.append(f"    Received: {turn.received}")
        lines.append(f"\nRationale: {r.rationale}")
        if r.failure_detail:
            lines.append(f"Failure detail: {r.failure_detail}")
    return "\n".join(lines)


def report(
    eval_results: list[EvalResult],
    description: str | None,
    config: Config,
    output_path: str | None,
) -> int:
    client = anthropic.Anthropic(api_key=config.qa_llm_api_key)
    narrative_prompt = _build_narrative_prompt(eval_results, description)

    with client.messages.stream(
        model=config.qa_llm.model,
        max_tokens=1024,
        thinking={"type": "adaptive"},
        system=_NARRATIVE_SYSTEM,
        messages=[{"role": "user", "content": narrative_prompt}],
    ) as stream:
        message = stream.get_final_message()

    narrative = next(
        (block.text for block in message.content if hasattr(block, "text")), ""
    )

    total = len(eval_results)
    passed = sum(1 for r in eval_results if r.passed)

    print(f"\n{'='*50}")
    print("QA AGENT REPORT")
    print(f"{'='*50}")
    print(f"\nResults: {passed}/{total} passed\n")
    print(narrative)
    print(f"\n{_format_results_table(eval_results)}")

    deep_dives = _format_failure_deep_dives(eval_results)
    if deep_dives:
        print(deep_dives)

    if output_path:
        json_data = {
            "summary": narrative,
            "pass_rate": passed / total if total else 0.0,
            "results": [
                {
                    "id": r.run_result.test_case.id,
                    "description": r.run_result.test_case.description,
                    "category": r.run_result.test_case.category,
                    "passed": r.passed,
                    "score": r.score,
                    "rationale": r.rationale,
                    "failure_detail": r.failure_detail,
                    "turns": [
                        {"sent": t.sent, "received": t.received}
                        for t in r.run_result.turns
                    ],
                }
                for r in eval_results
            ],
        }
        with open(output_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    return 0 if all(r.passed for r in eval_results) else 1
