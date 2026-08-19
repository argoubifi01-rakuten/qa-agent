import logging
import os
import sys

from qa_agent.llm import call_llm, call_llm_chat
from qa_agent.config import Config

logger = logging.getLogger(__name__)

_CONTEXT_FILENAME = "agent-context.md"

_ONBOARD_SYSTEM = """\
You are a QA analyst onboarding a new AI agent into an automated test pipeline.

You have been given the agent's system prompt and tool descriptions from the database.
Your job is to hold a brief conversation with the user to fill in details that the system \
prompt alone may not make clear: the agent's intended purpose, what good behaviour looks like, \
which edge cases matter, what it should refuse, etc.

Rules:
- Ask EXACTLY ONE question per reply.
- Questions should be specific and easy to answer.
- When you have enough to write a comprehensive QA context document (usually 3-5 questions), \
  output ONLY the token: READY_TO_SYNTHESIZE
- Never ask more than 7 questions total. If you have not finished by then, output \
  READY_TO_SYNTHESIZE anyway."""

_SYNTHESIS_SYSTEM = """\
You are a QA analyst. Using the agent's system prompt, tool descriptions, and the onboarding \
conversation, write a comprehensive QA context document in Markdown.

Structure:
# Agent: <name>

## Purpose
2-3 sentences describing what the agent is supposed to do.

## Domain
Primary domain and context (e.g. e-commerce, customer support, travel booking).

## Capabilities
Bullet list of specific things the agent can do.

## Tools
For each tool: its name, what it does, and when/how it is used.
If there are no tools, write "No tools defined."

## What to Test
Key happy-path behaviours, important edge cases, out-of-scope requests the agent should refuse, \
and adversarial / prompt-injection scenarios.

## Known Constraints
Any limitations, restrictions, or specific behaviours called out in the prompt or conversation.

Be concise. This document is consumed by an automated QA pipeline."""

_UPDATE_SYSTEM = """\
You are a QA analyst maintaining an agent context document.

You will be given:
- The current context document
- The advisor's proposed prompt changes (if any)
- A summary of the latest QA run

Update the context document to reflect what was learned: revised understanding of capabilities, \
corrected constraints, any new edge cases or failure patterns discovered, and what changes are \
being proposed. Keep the same Markdown structure. Return ONLY the updated document."""


def context_path(scenario_id: str, runs_dir: str = "runs") -> str:
    return os.path.join(runs_dir, scenario_id, _CONTEXT_FILENAME)


def load_context(scenario_id: str, runs_dir: str = "runs") -> str | None:
    path = context_path(scenario_id, runs_dir)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return f.read()


def save_context(scenario_id: str, markdown: str, runs_dir: str = "runs") -> str:
    sdir = os.path.join(runs_dir, scenario_id)
    os.makedirs(sdir, exist_ok=True)
    path = context_path(scenario_id, runs_dir)
    with open(path, "w") as f:
        f.write(markdown)
    logger.info("Agent context saved to %s", path)
    return path


def run_onboarding(
    scenario_id: str,
    agent_data: dict,
    config: Config,
    runs_dir: str = "runs",
    auto: bool = False,
) -> tuple[str, str]:
    """Onboarding that produces a QA context document.

    When auto=True (or stdin is not a TTY) the Q&A phase is skipped and the
    document is synthesised directly from the agent data.  Pass auto=True when
    calling from a non-interactive environment (CI, coding agent, etc.).
    """
    import sys

    name = agent_data.get("name", scenario_id)
    prompt = agent_data.get("prompt") or ""
    tools = agent_data.get("tools") or []

    tools_text = "\n".join(
        f"- **{t['name']}**: {t.get('description', '(no description)')}"
        for t in tools
    ) or "No tools defined."

    agent_info = (
        f"Agent name: {name}\n\n"
        f"System prompt:\n{prompt}\n\n"
        f"Tools:\n{tools_text}"
    )

    messages: list[dict] = [{"role": "user", "content": agent_info}]

    interactive = not auto and sys.stdin.isatty()

    if interactive:
        question_count = 0
        while True:
            response = call_llm_chat(
                system=_ONBOARD_SYSTEM,
                messages=messages,
                model=config.qa_llm.model,
                api_key=config.qa_llm_api_key,
                max_tokens=512,
            )

            if "READY_TO_SYNTHESIZE" in response:
                break

            question_count += 1
            print(f"\n  {response}\n")

            try:
                answer = input("  › ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            messages.append({"role": "assistant", "content": response})
            messages.append({"role": "user", "content": answer or "(no answer)"})

            if question_count >= 7:
                break

    # Build conversation transcript for synthesis
    qa_lines = []
    for msg in messages[1:]:
        prefix = "Q" if msg["role"] == "assistant" else "A"
        qa_lines.append(f"{prefix}: {msg['content']}")
    conversation_text = "\n".join(qa_lines) if qa_lines else "(no conversation)"

    synthesis_user = (
        f"{agent_info}\n\n"
        f"Onboarding conversation:\n{conversation_text}"
    )

    markdown = call_llm(
        system=_SYNTHESIS_SYSTEM,
        user=synthesis_user,
        model=config.qa_llm.model,
        api_key=config.qa_llm_api_key,
        max_tokens=2048,
    )

    path = save_context(scenario_id, markdown, runs_dir)
    return markdown, path


def generate_onboarding_questions(agent_data: dict, config: Config) -> list[str]:
    """Run the onboarding LLM loop with stub answers to collect all questions.

    Returns the list of questions the LLM would ask a human during interactive
    onboarding, so an external agent can present them in a chat UI and collect
    real answers before calling synthesize_from_qa().
    """
    name = agent_data.get("name", "Unknown")
    prompt = agent_data.get("prompt") or ""
    tools = agent_data.get("tools") or []

    tools_text = "\n".join(
        f"- **{t['name']}**: {t.get('description', '(no description)')}"
        for t in tools
    ) or "No tools defined."

    agent_info = (
        f"Agent name: {name}\n\n"
        f"System prompt:\n{prompt}\n\n"
        f"Tools:\n{tools_text}"
    )

    messages: list[dict] = [{"role": "user", "content": agent_info}]
    questions: list[str] = []

    for _ in range(7):
        response = call_llm_chat(
            system=_ONBOARD_SYSTEM,
            messages=messages,
            model=config.qa_llm.model,
            api_key=config.qa_llm_api_key,
            max_tokens=512,
        )
        if "READY_TO_SYNTHESIZE" in response:
            break
        questions.append(response.strip())
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": "(answer to be provided)"})

    return questions


def synthesize_from_qa(
    scenario_id: str,
    agent_data: dict,
    qa_pairs: list[dict],
    config: Config,
    runs_dir: str = "runs",
) -> tuple[str, str]:
    """Synthesise a QA context document from agent data and collected Q&A pairs.

    qa_pairs is a list of {"question": str, "answer": str} dicts.
    Returns (markdown, saved_path).
    """
    name = agent_data.get("name", scenario_id)
    prompt = agent_data.get("prompt") or ""
    tools = agent_data.get("tools") or []

    tools_text = "\n".join(
        f"- **{t['name']}**: {t.get('description', '(no description)')}"
        for t in tools
    ) or "No tools defined."

    agent_info = (
        f"Agent name: {name}\n\n"
        f"System prompt:\n{prompt}\n\n"
        f"Tools:\n{tools_text}"
    )

    conversation_text = "\n".join(
        f"Q: {pair['question']}\nA: {pair.get('answer', '(no answer)')}"
        for pair in qa_pairs
    ) or "(no conversation)"

    synthesis_user = (
        f"{agent_info}\n\n"
        f"Onboarding conversation:\n{conversation_text}"
    )

    markdown = call_llm(
        system=_SYNTHESIS_SYSTEM,
        user=synthesis_user,
        model=config.qa_llm.model,
        api_key=config.qa_llm_api_key,
        max_tokens=2048,
    )

    path = save_context(scenario_id, markdown, runs_dir)
    return markdown, path


def update_context(
    scenario_id: str,
    current_markdown: str,
    advice: str,
    run_summary: str,
    config: Config,
    runs_dir: str = "runs",
) -> str:
    """Regenerate the context markdown after a run. Returns updated markdown."""
    user_msg = (
        f"Current context document:\n{current_markdown}\n\n"
        f"Advisor proposed changes:\n{advice or 'None.'}\n\n"
        f"Latest run summary:\n{run_summary or 'No summary.'}"
    )
    updated = call_llm(
        system=_UPDATE_SYSTEM,
        user=user_msg,
        model=config.qa_llm.model,
        api_key=config.qa_llm_api_key,
        max_tokens=2048,
    )
    save_context(scenario_id, updated, runs_dir)
    return updated
