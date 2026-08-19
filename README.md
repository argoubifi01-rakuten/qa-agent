# qa-agent

An iterative QA pipeline for live AI agents. Generates test cases, runs them against a WebSocket endpoint, evaluates results with an LLM, and proposes prompt improvements — all from the command line.

Built for the Rakuten AI companion agent platform but designed to work with any agent that communicates over WebSocket and stores its data in Firestore.

---

## How it works

Each iteration does five things:

1. **Fetch** the agent's system prompt and tool definitions from Firestore
2. **Generate** test cases tailored to the agent's purpose (or use a static override file)
3. **Run** every test against the live WebSocket endpoint in parallel
4. **Evaluate** each result with an LLM judge against the test goal
5. **Advise** — a prompt advisor reviews the full failure history across all past iterations and proposes a revised prompt

Results are saved to `runs/<scenario_id>/run-<N>-<timestamp>.json`. A local dashboard lets you browse all runs, see pass-rate trends, and read prompt diffs.

---

## Requirements

- Python 3.11+
- Access to the target agent's WebSocket endpoint
- An OpenAI API key (used for the generator, evaluator, reporter, and advisor)
- GCP service account credentials with read access to the Firestore database that backs the agent platform

---

## Installation

```bash
git clone https://github.com/argoubifi01-rakuten/qa-agent.git
cd qa-agent
pip install -e .
```

If you need the legacy MongoDB path (requires VPN):

```bash
pip install -e ".[mongo]"
```

---

## Configuration

### 1. Environment variables

Copy `.env.example` to `.env` and fill in the required values:

```bash
cp .env.example .env
```

**Required:**

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI key used by the QA pipeline |
| `EVAL_MOCK_SECRET` | Sent as `X-Ninja-Eval-Mock` to bypass auth on test endpoints |

**Database — pick one:**

| Option | Variable | Notes |
|---|---|---|
| A1 (recommended) | `GCP_CREDENTIALS_BASE64` | Base64-encoded service account JSON. No file on disk, works anywhere. Generate: `base64 -i key.json \| tr -d '\n'` |
| A2 | `GOOGLE_APPLICATION_CREDENTIALS` | Path to a service account JSON key file |
| A3 | *(none)* | Application Default Credentials — run `gcloud auth application-default login` once |
| Legacy | `MONGODB_URI` | MongoDB wire-protocol URI. Requires VPN. Falls back to this if set. |

Optionally override the GCP project:
```
GOOGLE_CLOUD_PROJECT=your-project-id
```

### 2. `qa_agent.yaml`

This file controls which agent is being tested and how many tests to run:

```yaml
qa_llm:
  provider: openai
  model: gpt-4o          # model used for generation, evaluation, and advising

target:
  websocket_url: wss://your-agent-endpoint
  auth_url: https://your-agent-endpoint/api/v2/auth/anonymous
  thread_creation_url: https://your-agent-endpoint/api/v1/thread
  scenario_id: <scenario-id>
  wss_response_timeout: 120   # seconds before a test is retried

test_generation:
  num_test_cases: 20
  max_turns: 3

# Optional: seed tests from production traces instead of generating synthetically
trace_analyser:
  url: ""
```

---

## Onboarding an agent

Before running tests against a new agent, you need to create an `agent-context.md` — a short document that tells the evaluator what the agent is supposed to do, what it should refuse, and what good and bad responses look like.

```bash
qa-agent onboard --scenario-id <id>
```

This fetches the agent's system prompt and tool list from Firestore, then walks you through a short Q&A session. The resulting file is saved to `runs/<scenario_id>/agent-context.md`.

If you want to skip the Q&A and generate the context automatically from the agent data only:

```bash
qa-agent onboard --scenario-id <id> --auto
```

The context file is created automatically the first time you run `iterate` if it doesn't exist yet.

---

## Running tests

```bash
qa-agent iterate \
  --scenario-id <id> \
  --purpose "Plain-English description of what this agent should do"
```

Progress is printed as tests complete. At the end you get an **AGENT HANDOFF** block:

```
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  AGENT HANDOFF  — Iteration 3
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Scenario  : 6a5820569a280d2db9236800
  Pass rate : 14/20 (70%)  IMPROVING  [9/20 → 12/20 → 14/20]
  Run saved : runs/6a5820569a280d2db9236800/run-003-20260818-102030.json

  Advisor   : YES — proposed prompt available
  → Review the .advice field in the run file above
  → Apply it to Firestore, then run the next iteration

  Next step : qa-agent iterate --scenario-id 6a5820569a280d2db9236800 --purpose "..."
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### Iterative workflow

```
1. qa-agent iterate --scenario-id X --purpose "..."
2. Read the AGENT HANDOFF block
3. If the advisor proposed a new prompt:
     a. Open the run JSON → read the .advice field → find the PROPOSED PROMPT section
     b. Apply the new prompt to the agent in Firestore
4. Repeat from step 1
```

The advisor sees the full history of all previous iterations and won't repeat suggestions it has already made.

---

## Customising test cases

By default the generator creates test cases from the agent context. To override with a fixed set, create `test_cases_override.yaml` in the working directory:

```yaml
test_cases:
  - id: tc-s001
    description: "User asks for a product recommendation"
    category: general        # general | edge_case | out_of_scope | adversarial
    goal: "Agent should suggest relevant products and ask clarifying questions"
    input_message: "I'm looking for a good laptop under ¥100,000"
    follow_up_messages: null # null for single-turn; or a list of strings
    driver_instructions: null
```

Delete the file to go back to auto-generation. `follow_up_messages` and `driver_instructions` are mutually exclusive.

---

## Viewing history and the dashboard

```bash
# Print a pass-rate table for all past iterations
qa-agent history --scenario-id <id>

# Open the web dashboard in your browser
qa-agent dashboard
```

The dashboard shows pass-rate trends, per-test results, conversation turns, and prompt diffs across all scenarios and iterations.

---

## Claude Code skill

A `/qa-chat` skill is bundled in `.claude/skills/qa-chat.md`. When you open this project in Claude Code, you can type `/qa-chat` to enter an interactive QA session — designing tests, running iterations, and reading results conversationally without leaving the chat.

---

## Project structure

```
qa_agent/
├── cli.py                    # Entry point — all subcommands
├── config.py                 # Config loading from qa_agent.yaml + env
├── db.py                     # Firestore / MongoDB session abstraction
├── context.py                # Agent context (onboarding, Q&A, synthesis)
├── llm.py                    # OpenAI wrapper used by all pipeline stages
├── models.py                 # Shared data models (TestCase, RunResult, …)
├── storage.py                # Run file read/write
├── server.py                 # Dashboard HTTP server
├── signing.py                # Auth token signing
├── trace_analyser_client.py  # Production trace fetching (optional)
├── generator/                # Test case generation
├── runner/                   # WebSocket test runner
├── evaluator/                # LLM-based result evaluation
├── reporter/                 # Run report generation
└── advisor/                  # Prompt improvement advisor

.claude/skills/
├── qa-chat.md                # Claude Code conversational QA skill
└── qa-iterate.md             # Claude Code single-iteration skill

qa_agent.yaml                 # Active configuration
.env.example                  # Credential template
```

---

## Effective pass rate

The CLI and skill distinguish between **agent failures** and **runner errors** (network timeouts, empty WebSocket responses). Runner errors are excluded from the pass rate denominator so the number reflects actual agent quality, not infrastructure flakiness. The raw count and the exclusion reason are both reported.
