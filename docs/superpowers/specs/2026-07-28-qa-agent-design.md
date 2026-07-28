# QA Agent — Design Spec
**Date:** 2026-07-28

## Overview

A CLI tool that lets developers automatically test an LLM agent by generating test cases from its description and/or system prompt, running those cases against the live agent, and producing an evaluation report that narrates the overall experience and deep-dives on failures.

The system is designed to work as a local CLI today and be wrapped as a GitHub Action later without code changes.

---

## Goals

- Generate test cases automatically from an agent description and/or system prompt
- Execute test cases against a target agent via WebSocket (current transport)
- Evaluate responses using an LLM judge
- Report: narrative summary of overall quality + per-failure deep-dives
- Support single-turn and multi-turn conversations
- Be pluggable: QA LLM provider/model is configurable; target transport is swappable via adapter
- Never store secrets in files

---

## Architecture

Three pipeline stages, wired by a CLI entrypoint. Each stage communicates only through shared dataclasses (`models.py`).

```
config.yaml + env vars
        ↓
   cli.py  (argparse — loads config, accepts runtime flags)
        ↓
  Generator  →  list[TestCase]
        ↓
  Runner     →  list[RunResult]    (WebSocket calls)
        ↓
  Evaluator  →  list[EvalResult]   (LLM judge per result)
        ↓
  Reporter   →  stdout + optional JSON file
```

---

## Project Structure

```
qa_agent/
├── qa_agent/
│   ├── __init__.py
│   ├── cli.py                   # entrypoint: argparse, loads config, runs pipeline
│   ├── config.py                # loads qa_agent.yaml + env vars, validates, exposes typed config
│   ├── models.py                # shared dataclasses: TestCase, Turn, RunResult, EvalResult
│   ├── generator/
│   │   ├── __init__.py
│   │   └── generator.py         # description + prompt → list[TestCase]
│   ├── runner/
│   │   ├── __init__.py
│   │   ├── base.py              # abstract Runner interface
│   │   └── websocket_runner.py  # WebSocket + HTTP auth adapter
│   ├── evaluator/
│   │   ├── __init__.py
│   │   └── evaluator.py         # RunResult → EvalResult (LLM call per case)
│   └── reporter/
│       ├── __init__.py
│       └── reporter.py          # list[EvalResult] → narrative report + JSON
├── qa_agent.yaml                # non-sensitive config
├── .env.example                 # documents required env vars
├── pyproject.toml
└── README.md
```

---

## Configuration

### `qa_agent.yaml` (checked into repo — no secrets)

```yaml
qa_llm:
  provider: anthropic       # or: openai
  model: claude-sonnet-5

target:
  websocket_url: wss://your-backend/ws
  auth_url: https://your-backend/api/v2/auth/anonymous
  thread_creation_url: https://your-backend/api/v2/threads
  wss_response_timeout: 120

test_generation:
  num_test_cases: 10
  max_turns: 1              # set to 3+ for multi-turn conversations
```

### Environment variables (never in YAML)

| Variable | Purpose |
|---|---|
| `QA_LLM_API_KEY` | API key for the QA LLM provider |
| `TARGET_AUTH_SECRET` | Auth secret for the target agent |
| `MONGODB_URI` | MongoDB credentials (reserved for future use) |

### CLI flags (per-run identity)

```bash
qa-agent run \
  --agent-description "Customer support agent for e-commerce" \
  --prompt-file ./prompts/system.txt \
  --env dev \
  --output report.json
```

`--agent-description` and `--prompt-file` are both optional but at least one must be provided. If neither is given, `cli.py` exits immediately with a clear error message before any LLM call is made.

---

## Data Models (`models.py`)

```python
@dataclass
class TestCase:
    id: str
    description: str        # what this case tests
    category: str           # "general", "edge_case", "out_of_scope", "adversarial"
    goal: str               # full conversation goal (used by runner in multi-turn mode)
    input_message: str      # opening message sent to the agent

@dataclass
class Turn:
    sent: str               # message sent by QA agent
    received: str           # response from target agent

@dataclass
class RunResult:
    test_case: TestCase
    turns: list[Turn]       # always a list; single-turn has exactly one entry
    success: bool           # False if timeout, connection error, etc.
    error: str | None

@dataclass
class EvalResult:
    run_result: RunResult
    passed: bool
    score: float            # 0.0–1.0
    rationale: str
    failure_detail: str | None   # populated only when passed=False
```

---

## Pipeline Stage Details

### Generator

- Input: agent description (str | None), system prompt (str | None), config
- Makes one LLM call instructing it to produce `num_test_cases` test cases as structured JSON
- LLM is prompted to cover four categories:
  - `general` — happy-path, expected use cases
  - `edge_case` — empty input, very long input, ambiguous intent
  - `out_of_scope` — requests the agent should refuse or redirect
  - `adversarial` — prompt injection, conflicting instructions
- Validates JSON response against TestCase schema; retries once on malformed output, then raises

### Runner (`websocket_runner.py`)

- For each TestCase:
  1. HTTP POST to `auth_url` → `accessToken`
  2. HTTP POST to `thread_creation_url` → `thread_id`
  3. Open WebSocket connection
  4. Send opening message (`test_case.input_message`)
  5. Stream responses until `chatResponseStatus == "DONE"`, collect full response text
  6. **If `max_turns == 1`:** close connection, return `RunResult` with one `Turn`
  7. **If `max_turns > 1`:** call QA LLM with conversation history + goal to decide next message or signal `DONE`; repeat from step 4 until LLM signals done or `max_turns` reached; close connection
- On timeout or connection error: `RunResult(success=False, error=...)`

The `base.py` abstract interface defines `run(test_case: TestCase) -> RunResult` — future transports (HTTP, CLI) implement this same interface.

### Evaluator

- Input: `RunResult`, agent description/prompt (for context)
- If `run_result.success == False`: marks as failed automatically, no LLM call
- Otherwise: one LLM call per result, passing:
  - Agent description and/or system prompt
  - TestCase category and goal
  - Full conversation (all turns)
- LLM returns structured verdict: `passed`, `score`, `rationale`, `failure_detail`

### Reporter

- Input: `list[EvalResult]`, agent description
- Two parts:
  1. **Narrative summary** — one LLM call: overall pass rate, patterns in failures, general quality assessment
  2. **Results table** — every test case as one line (id, category, passed/failed, score)
  3. **Failure deep-dives** — for each failed case: input messages, responses, rationale, failure_detail. Passing cases are not expanded.
- Stdout by default
- If `--output report.json`: also writes machine-readable JSON with all `EvalResult` objects

---

## Error Handling

- Missing required env vars → config validation fails at startup with a clear message listing what's missing
- LLM call failure in Generator → raises, run aborted
- WebSocket timeout → `RunResult(success=False)`, evaluation marks it failed, run continues
- LLM call failure in Evaluator → marks that result as failed with `error` reason, run continues
- Exit code: 0 if all cases pass, 1 if any case fails (enables CI use)

---

## Future: GitHub Action

When ready, a thin `action.yml` maps Action `inputs:` to CLI flags and Action `secrets:` to env vars. The core code is unchanged. Example:

```yaml
# action.yml
inputs:
  agent_description:
    description: 'Description of the agent under test'
  prompt_file:
    description: 'Path to system prompt file'
  env:
    description: 'Target environment'
    default: 'dev'
```

```yaml
# .github/workflows/qa.yml usage
- uses: ./
  with:
    agent_description: "Customer support agent"
    prompt_file: ./prompts/system.txt
    env: dev
  env:
    QA_LLM_API_KEY: ${{ secrets.QA_LLM_API_KEY }}
    TARGET_AUTH_SECRET: ${{ secrets.TARGET_AUTH_SECRET }}
```

---

## Out of Scope (v1)

- MongoDB tool-call inspection (evaluates final text responses only)
- Non-WebSocket transports (HTTP REST, Python function, CLI subprocess)
- Web UI or VS Code extension
- Automatic test case diffing between runs (prompt change regression tracking)
