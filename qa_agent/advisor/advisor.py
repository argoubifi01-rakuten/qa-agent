import logging
from qa_agent.llm import call_llm
from qa_agent.models import EvalResult
from qa_agent.config import Config

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a prompt engineering expert. Given:
- The stated purpose of an AI agent
- The agent's current system prompt
- QA evaluation results
- Optionally: a unified diff showing what changed from the baseline prompt to the current one
- Optionally: a history of previous QA iterations for this scenario

When iteration history is provided, treat it as a learning record — what changes were tried before,
what worked, what didn't. Avoid proposing changes that were already tried and didn't improve results.
Factor the trend (improving, regressing, stable) into the confidence and tone of your assessment.

When a diff is provided, use it to understand the intent behind the changes — what behaviour \
was being added, removed, or corrected. Factor this into your assessment: did the changes move \
the agent closer to or further from the stated purpose? Do the QA failures suggest the changes \
introduced regressions, or are they unrelated?

Return your response in this format:
---
ASSESSMENT
<2-3 sentences: does the prompt serve the stated purpose? What's missing or misaligned?>

CHANGE ANALYSIS
<If a diff was provided: 2-3 sentences on what the prompt changes were trying to achieve and \
whether the QA results confirm they worked. Omit this section if no diff was provided.>

PROPOSED PROMPT
<the full improved system prompt, or "No changes needed." if it already serves the purpose well>
---"""


def _format_iteration_history(history: list[dict]) -> str:
    if not history:
        return ""
    lines = ["\nIteration history for this scenario (oldest → newest):"]
    for run in history:
        n = run.get("iteration", "?")
        ts = (run.get("timestamp") or "")[:10]
        results = run.get("results", [])
        total = len(results)
        passed = sum(1 for r in results if r.get("passed"))
        pct = int(passed / total * 100) if total else 0
        lines.append(f"\n  Iteration {n} ({ts}): {passed}/{total} passed ({pct}%)")
        if run.get("summary"):
            lines.append(f"    Summary: {run['summary'][:300]}")
        advice = run.get("advice") or ""
        if "PROPOSED PROMPT" in advice:
            proposed = advice.split("PROPOSED PROMPT", 1)[-1].strip()
            if proposed and proposed.lower() not in ("no changes needed.", "no changes needed"):
                preview = proposed[:250].replace("\n", " ")
                lines.append(f"    Proposed prompt (excerpt): {preview}…")
            else:
                lines.append("    Proposed prompt: No changes needed.")
    return "\n".join(lines)


def _build_advisor_prompt(
    current_prompt: str,
    eval_results: list[EvalResult],
    agent_context: str | None = None,
    iteration_history: list[dict] | None = None,
) -> str:
    total = len(eval_results)
    passed = sum(1 for r in eval_results if r.passed)
    failures = [r for r in eval_results if not r.passed]

    parts = [f"QA results: {passed}/{total} passed"]

    if agent_context:
        parts.append(f"\nAgent context document:\n{agent_context}")

    if failures:
        parts.append("\nFailing cases:")
        for r in failures:
            parts.append(
                f"  [{r.run_result.test_case.category}] {r.run_result.test_case.description}"
                f"\n  Rationale: {r.rationale}"
                + (f"\n  Detail: {r.failure_detail}" if r.failure_detail else "")
            )

    if iteration_history:
        parts.append(_format_iteration_history(iteration_history))

    parts.append(f"\nCurrent system prompt:\n{current_prompt}")
    return "\n".join(parts)


def advise(
    current_prompt: str,
    eval_results: list[EvalResult],
    config: Config,
    agent_context: str | None = None,
    iteration_history: list[dict] | None = None,
) -> str:
    """Return an advisory string with assessment and optionally a proposed prompt."""
    logger.info("Running prompt advisor")
    user_msg = _build_advisor_prompt(
        current_prompt, eval_results, agent_context, iteration_history
    )
    result = call_llm(
        system=_SYSTEM_PROMPT,
        user=user_msg,
        model=config.qa_llm.model,
        api_key=config.qa_llm_api_key,
        max_tokens=2048,
    )
    logger.debug("Advisor response length=%d chars", len(result))
    return result
