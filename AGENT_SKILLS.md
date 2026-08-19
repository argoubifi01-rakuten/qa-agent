# QA Agent — Iterative Evaluation Skills

This document tells any coding agent (Claude Code, Cursor, Copilot, etc.) how to run iterative
QA evaluations against a Rakuten AI scenario agent. Read it before issuing any `qa-agent` commands.

---

## What this tool does

`qa-agent` evaluates a live Rakuten AI companion agent by:
1. Fetching its system prompt from MongoDB
2. Generating test cases tailored to its stated purpose
3. Running those tests against the live WebSocket endpoint
4. Judging results with an LLM evaluator
5. Running a prompt advisor that sees all previous iterations and proposes an improved prompt
6. Printing a **handoff block** that tells you exactly what to do next

Each run is saved under `runs/<scenario_id>/run-<N>-<timestamp>.json`.

---

## Commands

### Run one iteration (primary command for iterative use)

```sh
qa-agent iterate \
  --scenario-id <SCENARIO_ID> \
  --purpose "<plain-English description of what the agent should do>" \
  [--config qa_agent.yaml] \
  [--baseline-id <BASELINE_SCENARIO_ID>] \
  [--runs-dir runs] \
  [--verbose]
```

- Always non-interactive, no browser
- Prints a full **AGENT HANDOFF** block at the end (see below)
- Use this for all scripted / automated iterations

### Show iteration history for a scenario

```sh
qa-agent history --scenario-id <SCENARIO_ID> [--runs-dir runs]
```

Prints a table of all past iterations: date, pass rate, status (improving / stable / regressing).

### Open the dashboard (browse all past runs)

```sh
qa-agent dashboard [--port 8080] [--runs-dir runs]
```

Starts the local HTTP server **and** opens your browser automatically at `http://localhost:8080`.
Shows all runs across all scenarios: pass-rate bars, prompt diffs, per-test conversation turns,
and the advisor section. Press Ctrl+C to stop.

```sh
qa-agent serve [--port 8080] [--runs-dir runs]
```

Same as `dashboard` but does **not** open the browser — useful when running headlessly.

### Run with browser dashboard (interactive use)

```sh
qa-agent run \
  --scenario-id <SCENARIO_ID> \
  --purpose "<description>" \
  [--no-browser]
```

With `--no-browser`, same as `iterate` but using the `run` subcommand.
Without `--no-browser`, opens a dashboard at `http://localhost:8080` after the run.

### Bare interactive mode

```sh
qa-agent
```

Prompts for scenario ID and purpose, then runs the full pipeline and opens the dashboard.

---

## The iterative workflow

```
┌────────────────────────────────────────────────────────────────┐
│  1. Run:  qa-agent iterate --scenario-id X --purpose "..."     │
│  2. Read the AGENT HANDOFF block printed at the end            │
│  3. If advisor proposed a new prompt:                          │
│       a. Read runs/<scenario_id>/run-<N>-*.json (.advice field)│
│       b. Extract the PROPOSED PROMPT section                   │
│       c. Apply it to MongoDB (task_agents.instructions.prompt) │
│  4. Repeat from step 1                                         │
└────────────────────────────────────────────────────────────────┘
```

You decide when to apply a proposed prompt and when to run the next iteration.
The advisor sees the full history of previous iterations and will not repeat
suggestions that have already been tried.

---

## Reading the AGENT HANDOFF block

At the end of every `qa-agent iterate` run, you will see:

```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AGENT HANDOFF  — Iteration 3
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scenario  : 6a5820569a280d2db9236800
  Pass rate : 14/20 (70%)  IMPROVING  [9/20 → 12/20 → 14/20]
  Run saved : runs/6a5820569a280d2db9236800/run-003-20260818-102030.json

  Advisor   : YES — proposed prompt available
  → Review the .advice field in the run file above
  → Apply it to MongoDB, then run the next iteration

  Next step : qa-agent iterate --scenario-id 6a5820569a280d2db9236800 --purpose "..."
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

| Field | Meaning |
|---|---|
| `Iteration` | Which iteration this is (1-based, per scenario) |
| `Pass rate` | `passed/total (%)` — higher is better |
| `IMPROVING / STABLE / REGRESSING / FIRST_RUN` | Trend vs previous iteration |
| `[9/20 → ...]` | Full pass-rate history for this scenario |
| `Advisor: YES` | The advisor produced a proposed prompt — review and apply it |
| `Next step` | The exact command to run next iteration |

---

## Where run data lives

```
runs/
└── <scenario_id>/
    ├── run-001-20260814-082301.json   ← iteration 1
    ├── run-002-20260815-140512.json   ← iteration 2
    └── run-003-20260818-102030.json   ← iteration 3
```

Each JSON file contains:
- `iteration` — iteration number
- `timestamp` — ISO 8601 UTC
- `scenario_id`, `purpose`
- `pass_rate` — float 0–1
- `summary` — LLM narrative
- `advice` — full advisor output; look for `PROPOSED PROMPT` section
- `prompt_comparison` — diff vs baseline (if applicable)
- `results[]` — per-test-case: `id`, `description`, `category`, `passed`, `score`, `rationale`, `failure_detail`, `turns[]`

---

## Prerequisites

- Python package installed: `pip install -e .` (from the `qa_agent/` directory)
- `.env` file with `OPENAI_API_KEY`, `EVAL_MOCK_SECRET`, `MONGODB_URI`
- `qa_agent.yaml` config in the working directory (or pass `--config <path>`)
