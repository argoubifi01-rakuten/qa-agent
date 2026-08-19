import os
import sys
import time
import threading
import webbrowser
import logging
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from qa_agent.config import load_config
from qa_agent.server import serve
from qa_agent.db import fetch_system_prompt, fetch_agent_data, _DEFAULT_BASELINE_ID
from qa_agent.context import (
    load_context, save_context, run_onboarding, update_context, context_path,
    generate_onboarding_questions, synthesize_from_qa,
)
from qa_agent.generator.generator import generate_test_cases
from qa_agent.runner.websocket_runner import WebSocketRunner
from qa_agent.evaluator.evaluator import evaluate
from qa_agent.reporter.reporter import report
from qa_agent.advisor.advisor import advise
from qa_agent.storage import save_run, load_scenario_runs, get_iteration_number


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger(__name__)

# ── terminal helpers ──────────────────────────────────────────────────────────

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_GREEN  = "\033[32m"
_RED    = "\033[31m"
_YELLOW = "\033[33m"
_CYAN   = "\033[36m"
_BLUE   = "\033[34m"


def _c(color: str, text: str) -> str:
    return f"{color}{text}{_RESET}"


def _banner() -> None:
    print()
    print(_c(_BOLD, "  ┌─────────────────────────────────────┐"))
    print(_c(_BOLD, "  │") + _c(_CYAN + _BOLD, "         Rakuten AI · QA Agent       ") + _c(_BOLD, "│"))
    print(_c(_BOLD, "  └─────────────────────────────────────┘"))
    print()
    print(_c(_DIM, "  Evaluates a live AI agent and opens results in your browser."))
    print()


def _step(label: str) -> None:
    print(f"\n  {_c(_BOLD, '▸')} {label}", flush=True)


def _ok(label: str) -> None:
    print(f"  {_c(_GREEN, '✓')} {label}", flush=True)


def _warn(label: str) -> None:
    print(f"  {_c(_YELLOW, '!')} {label}", flush=True)


def _err(label: str) -> None:
    print(f"  {_c(_RED, '✗')} {label}", flush=True)


def _prompt(question: str, default: str | None = None, validator=None) -> str:
    hint = f"  [{_c(_DIM, default)}]" if default else ""
    while True:
        try:
            val = input(f"  {_c(_BOLD, question)}{hint}\n  {_c(_CYAN, '›')} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if not val and default is not None:
            return default
        if val:
            if validator:
                err = validator(val)
                if err:
                    _err(err)
                    continue
            return val
        _err("This field is required.")


# ── interactive prompt ────────────────────────────────────────────────────────

def _interactive_prompt() -> tuple[str, str]:
    _banner()

    print(_c(_DIM, "  ─" * 22))
    print()
    print(f"  {_c(_BOLD, 'Scenario ID')}")
    print(_c(_DIM, "  The ID of the agent version you want to evaluate."))
    scenario_id = _prompt("Paste the scenario ID")

    print()
    print(f"  {_c(_BOLD, 'Purpose')}")
    print(_c(_DIM, "  Describe in plain English what this agent is supposed to do."))
    print(_c(_DIM, "  Example: Shopping agent that shows a clarification form for vague queries."))
    purpose = _prompt("What does this agent do?")

    print()
    print(_c(_DIM, "  ─" * 22))
    return scenario_id, purpose


# ── handoff block ─────────────────────────────────────────────────────────────

def _print_handoff(
    scenario_id: str,
    iteration: int,
    passed: int,
    total: int,
    prior_runs: list[dict],
    saved_path: str,
    purpose: str,
    has_proposed_prompt: bool,
) -> None:
    pct = int(passed / total * 100) if total else 0
    curr_rate = passed / total if total else 0.0

    if not prior_runs:
        status = "FIRST_RUN"
        trend_str = f"{passed}/{total}"
    else:
        prev_rate = prior_runs[-1].get("pass_rate", 0.0)
        if curr_rate > prev_rate + 0.05:
            status = "IMPROVING"
        elif curr_rate < prev_rate - 0.05:
            status = "REGRESSING"
        else:
            status = "STABLE"

        def _rate_label(run: dict) -> str:
            results = run.get("results", [])
            n = len(results)
            p = sum(1 for r in results if r.get("passed"))
            return f"{p}/{n}"

        labels = [_rate_label(r) for r in prior_runs] + [f"{passed}/{total}"]
        trend_str = " → ".join(labels)

    status_color = {
        "IMPROVING": _GREEN, "FIRST_RUN": _CYAN,
        "STABLE": _YELLOW,   "REGRESSING": _RED,
    }.get(status, _RESET)

    bar = "━" * 50
    print(f"\n  {_c(_DIM, bar)}")
    print(f"  {_c(_BOLD, f'AGENT HANDOFF')}  {_c(_DIM, f'— Iteration {iteration}')}")
    print(f"  {_c(_DIM, bar)}")
    print(f"  Scenario  : {scenario_id}")
    print(f"  Pass rate : {_c(_BOLD, f'{passed}/{total} ({pct}%)')}  "
          f"{_c(status_color, status)}"
          + (f"  {_c(_DIM, trend_str)}" if len(prior_runs) > 0 else ""))
    print(f"  Run saved : {_c(_DIM, saved_path)}")

    if has_proposed_prompt:
        print(f"\n  Advisor   : {_c(_GREEN, 'YES')} — proposed prompt available")
        print(f"  {_c(_DIM, '→ Review the .advice field in the run file above')}")
        print(f"  {_c(_DIM, '→ Apply it to MongoDB, then run the next iteration')}")
    else:
        print(f"\n  Advisor   : {_c(_DIM, 'No changes suggested')}")

    purpose_escaped = purpose.replace('"', '\\"')
    print(f"\n  Next step : {_c(_BOLD, 'qa-agent iterate')} "
          f"--scenario-id {scenario_id} "
          f'--purpose "{purpose_escaped}"')
    print(f"  {_c(_DIM, bar)}\n")


# ── run pipeline ──────────────────────────────────────────────────────────────

def _ask_yes_no(question: str) -> bool:
    """Prompt the user for a yes/no answer. Returns True for yes."""
    try:
        val = input(f"  {_c(_BOLD, question)} {_c(_DIM, '[y/N]')} ").strip().lower()
        return val in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        print()
        return False


def _run(scenario_id: str, purpose: str, config_path: str,
         runs_dir: str, verbose: bool,
         no_browser: bool = False) -> int:

    _configure_logging(verbose)

    try:
        config = load_config(config_path)
    except (FileNotFoundError, EnvironmentError) as exc:
        _err(str(exc))
        return 1

    config.target.scenario_id = scenario_id

    # Load iteration history for this scenario before running
    prior_runs = load_scenario_runs(scenario_id, runs_dir)
    iteration = len(prior_runs) + 1

    if iteration > 1:
        _ok(f"Continuing from iteration {iteration - 1}  "
            f"({_c(_DIM, f'{len(prior_runs)} prior run(s) loaded')})")

    # ── agent context ─────────────────────────────────────────────────────────
    agent_context = load_context(scenario_id, runs_dir)
    if agent_context is None:
        _warn("No agent context found — starting onboarding…")
        print()
        agent_data = fetch_agent_data(scenario_id=scenario_id, mongodb_uri=config.mongodb_uri)
        agent_context, ctx_path = run_onboarding(
            scenario_id=scenario_id,
            agent_data=agent_data,
            config=config,
            runs_dir=runs_dir,
            auto=True,
        )
        print()
        _ok(f"Context saved → {_c(_DIM, ctx_path)}")
    else:
        _ok(f"Agent context loaded ({len(agent_context):,} chars)")

    # ── fetch raw prompt (advisor only) ───────────────────────────────────────
    _step("Fetching agent prompt for advisor…")
    current_prompt = fetch_system_prompt(
        scenario_id=scenario_id,
        mongodb_uri=config.mongodb_uri,
    )
    if current_prompt:
        _ok(f"Prompt loaded ({len(current_prompt):,} chars)")
    else:
        _warn("No prompt found — advisor will be skipped")

    # ── generate test cases ───────────────────────────────────────────────────
    n = config.test_generation.num_test_cases
    _step(f"Generating {n} test cases…")
    test_cases = generate_test_cases(
        agent_context=agent_context,
        config=config,
    )
    _ok(f"{len(test_cases)} test cases ready")

    # ── run tests in parallel ─────────────────────────────────────────────────
    n_tests = len(test_cases)
    _step(f"Running {n_tests} tests in parallel (up to {config.test_generation.max_turns} turns each)…")
    print()
    runner = WebSocketRunner(config=config, description=purpose, prompt=current_prompt)
    run_results = [None] * n_tests
    _output_lock = threading.Lock()

    def _run_one(idx: int, tc) -> tuple[int, object]:
        result = runner.run(tc)
        prefix = f"  [{idx + 1:>2}/{n_tests}]  {_c(_DIM, tc.id)}  "
        with _output_lock:
            if result.success:
                turns_label = _c(_DIM, f"  ({len(result.turns)}t)")
                print(f"{prefix}{_c(_GREEN, '●')}{turns_label}  {tc.description[:55]}", flush=True)
            else:
                print(f"{prefix}{_c(_RED, '●')}  {tc.description[:55]}", flush=True)
        return idx, result

    workers = min(n_tests, 10)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_run_one, i, tc) for i, tc in enumerate(test_cases)]
        for f in as_completed(futures):
            idx, result = f.result()
            run_results[idx] = result

    # ── evaluate ──────────────────────────────────────────────────────────────
    print()
    _step("Evaluating results…")
    eval_results = evaluate(
        run_results=run_results,
        agent_context=agent_context,
        config=config,
    )
    passed = sum(1 for r in eval_results if r.passed)
    total  = len(eval_results)
    color  = _GREEN if passed == total else (_YELLOW if passed >= total // 2 else _RED)
    _ok(f"{_c(color + _BOLD, f'{passed}/{total} passed')}  ({int(passed/total*100)}%)")

    for r in eval_results:
        icon = _c(_GREEN, "✓") if r.passed else _c(_RED, "✗")
        print(f"       {icon}  {_c(_DIM, r.run_result.test_case.id)}  {r.run_result.test_case.description[:60]}")

    # ── advisor ───────────────────────────────────────────────────────────────
    advice = None
    if current_prompt:
        _step("Running prompt advisor…")
        advice = advise(
            current_prompt=current_prompt,
            eval_results=eval_results,
            config=config,
            agent_context=agent_context,
            iteration_history=prior_runs if prior_runs else None,
        )
        _ok("Advisor recommendations ready")

    # ── report + save ─────────────────────────────────────────────────────────
    exit_code, run_data = report(
        eval_results=eval_results,
        description=purpose,
        config=config,
        scenario_id=scenario_id,
        advice=advice,
        agent_context=agent_context,
    )
    run_data["iteration"] = iteration
    saved_path = save_run(run_data, runs_dir=runs_dir)
    _ok(f"Run saved → {_c(_DIM, saved_path)}")

    # ── post-run context update ───────────────────────────────────────────────
    has_proposed = bool(
        advice and "PROPOSED PROMPT" in advice
        and "No changes needed" not in advice.split("PROPOSED PROMPT", 1)[-1][:50]
    )
    if has_proposed:
        print()
        if _ask_yes_no("Advisor proposed prompt changes. Update agent context?"):
            _step("Updating agent context…")
            summary = run_data.get("summary", "")
            update_context(
                scenario_id=scenario_id,
                current_markdown=agent_context,
                advice=advice,
                run_summary=summary,
                config=config,
                runs_dir=runs_dir,
            )
            _ok(f"Context updated → {_c(_DIM, context_path(scenario_id, runs_dir))}")

    # ── agent handoff block ───────────────────────────────────────────────────
    _print_handoff(scenario_id, iteration, passed, total,
                   prior_runs, saved_path, purpose, has_proposed)

    if no_browser:
        return exit_code

    # ── open dashboard ────────────────────────────────────────────────────────
    port = 8080
    _step(f"Opening dashboard at {_c(_CYAN + _BOLD, f'http://localhost:{port}')} …")

    server_thread = threading.Thread(
        target=serve, kwargs={"port": port, "runs_dir": runs_dir, "quiet": True}, daemon=True
    )
    server_thread.start()
    time.sleep(0.4)
    webbrowser.open(f"http://localhost:{port}")

    print()
    print(_c(_DIM, "  ─" * 22))
    print(f"  {_c(_BOLD, 'Results are live.')}  Press {_c(_BOLD, 'Ctrl+C')} to stop the dashboard.")
    print(_c(_DIM, "  ─" * 22))
    print()

    try:
        server_thread.join()
    except KeyboardInterrupt:
        print(f"\n  {_c(_DIM, 'Dashboard stopped.')}")

    return exit_code


# ── history command ───────────────────────────────────────────────────────────

def _history(scenario_id: str, runs_dir: str) -> int:
    runs = load_scenario_runs(scenario_id, runs_dir)
    if not runs:
        print(f"No runs found for scenario {scenario_id} in {runs_dir}/")
        return 0

    print(f"\n  Iteration history — scenario {scenario_id}")
    print(f"  {'─' * 60}")
    print(f"  {'#':<5} {'Date':<12} {'Pass rate':<12} {'Status':<12} {'File'}")
    print(f"  {'─' * 60}")

    prev_rate: float | None = None
    for run in runs:
        n = run.get("iteration", "?")
        ts = (run.get("timestamp") or "")[:10]
        results = run.get("results", [])
        total = len(results)
        passed_count = sum(1 for r in results if r.get("passed"))
        rate = passed_count / total if total else 0.0
        pct = int(rate * 100)
        rate_str = f"{passed_count}/{total} ({pct}%)"

        if prev_rate is None:
            status = "first"
        elif rate > prev_rate + 0.05:
            status = "improving"
        elif rate < prev_rate - 0.05:
            status = "regressing"
        else:
            status = "stable"
        prev_rate = rate

        fname = run.get("_filename", "")
        print(f"  {str(n):<5} {ts:<12} {rate_str:<12} {status:<12} {fname}")

    print(f"  {'─' * 60}")
    print(f"  {len(runs)} iteration(s) total\n")
    return 0


# ── onboard command ───────────────────────────────────────────────────────────

def _onboard(scenario_id: str, config_path: str, runs_dir: str, verbose: bool, auto: bool = False) -> int:
    _configure_logging(verbose)

    try:
        config = load_config(config_path)
    except (FileNotFoundError, EnvironmentError) as exc:
        _err(str(exc))
        return 1

    existing = load_context(scenario_id, runs_dir)
    if existing is not None and not auto:
        print()
        if not _ask_yes_no(
            f"Agent context already exists for {scenario_id}. Re-run onboarding and overwrite?"
        ):
            _ok("Keeping existing context.")
            return 0
        print()

    _step("Fetching agent data from database…")
    agent_data = fetch_agent_data(scenario_id=scenario_id, mongodb_uri=config.mongodb_uri)
    name = agent_data.get("name", scenario_id)
    tools = agent_data.get("tools") or []
    _ok(f"Agent: {name}  |  {len(tools)} tool(s)")

    print()
    _step("Starting onboarding conversation…")
    print(_c(_DIM, "  Answer the questions below to help build the agent context document."))
    print()

    _, ctx_path = run_onboarding(
        scenario_id=scenario_id,
        agent_data=agent_data,
        config=config,
        runs_dir=runs_dir,
        auto=auto,
    )

    print()
    _ok(f"Agent context saved → {_c(_DIM, ctx_path)}")
    return 0


# ── onboard-questions command ─────────────────────────────────────────────────

def _onboard_questions(scenario_id: str, config_path: str, runs_dir: str, verbose: bool) -> int:
    import json
    _configure_logging(verbose)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, EnvironmentError) as exc:
        _err(str(exc))
        return 1

    _step("Fetching agent data from database…")
    agent_data = fetch_agent_data(scenario_id=scenario_id, mongodb_uri=config.mongodb_uri)
    tools = agent_data.get("tools") or []
    _ok(f"Agent: {agent_data.get('name', scenario_id)}  |  {len(tools)} tool(s)")

    _step("Generating onboarding questions…")
    questions = generate_onboarding_questions(agent_data=agent_data, config=config)
    _ok(f"{len(questions)} question(s) generated")

    print(json.dumps(questions, ensure_ascii=False, indent=2))
    return 0


# ── onboard-synthesize command ────────────────────────────────────────────────

def _onboard_synthesize(
    scenario_id: str, qa_json: str, config_path: str, runs_dir: str, verbose: bool
) -> int:
    import json
    _configure_logging(verbose)
    try:
        config = load_config(config_path)
    except (FileNotFoundError, EnvironmentError) as exc:
        _err(str(exc))
        return 1

    try:
        qa_pairs = json.loads(qa_json)
    except json.JSONDecodeError as exc:
        _err(f"Invalid --qa JSON: {exc}")
        return 1

    _step("Fetching agent data from database…")
    agent_data = fetch_agent_data(scenario_id=scenario_id, mongodb_uri=config.mongodb_uri)
    _ok(f"Agent: {agent_data.get('name', scenario_id)}")

    _step("Synthesising agent context document…")
    _, ctx_path = synthesize_from_qa(
        scenario_id=scenario_id,
        agent_data=agent_data,
        qa_pairs=qa_pairs,
        config=config,
        runs_dir=runs_dir,
    )
    _ok(f"Agent context saved → {_c(_DIM, ctx_path)}")
    return 0


# ── entry point ───────────────────────────────────────────────────────────────

def _ensure_ssl_certs() -> None:
    """Set SSL_CERT_FILE to the certifi bundle when not already configured.

    Needed on macOS/pyenv which lack system CA certs for PyMongo TLS.
    The Firestore SDK uses google-auth which bundles its own certs, so this
    only matters for the legacy MONGODB_URI path.
    """
    if os.environ.get("SSL_CERT_FILE"):
        return
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass


def main() -> None:
    # Ensure output is flushed immediately so progress is visible in all terminals/pipes.
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)

    load_dotenv()
    _ensure_ssl_certs()

    parser = argparse.ArgumentParser(
        prog="qa-agent",
        description="Evaluate a live AI agent. Run with no arguments for interactive mode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── onboard ───────────────────────────────────────────────────────────────
    onboard_parser = subparsers.add_parser(
        "onboard",
        help="Onboard a new scenario agent (creates agent-context.md via Q&A)",
    )
    onboard_parser.add_argument("--scenario-id", required=True, type=str)
    onboard_parser.add_argument("--config", type=str, default="qa_agent.yaml")
    onboard_parser.add_argument("--runs-dir", type=str, default="runs")
    onboard_parser.add_argument("--verbose", "-v", action="store_true")
    onboard_parser.add_argument(
        "--auto", action="store_true",
        help="Skip interactive Q&A and synthesise context from agent data only",
    )

    # ── run ───────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser("run", help="Run QA evaluation (non-interactive)")
    run_parser.add_argument("--scenario-id", type=str, default=None)
    run_parser.add_argument("--purpose", type=str, default=None)
    run_parser.add_argument("--config", type=str, default="qa_agent.yaml")
    run_parser.add_argument("--runs-dir", type=str, default="runs")
    run_parser.add_argument("--verbose", "-v", action="store_true")
    run_parser.add_argument(
        "--no-browser", action="store_true",
        help="Save results but do not open the dashboard",
    )

    # ── iterate ───────────────────────────────────────────────────────────────
    iter_parser = subparsers.add_parser(
        "iterate",
        help="Run one QA iteration (non-interactive, no browser, prints agent handoff)",
    )
    iter_parser.add_argument("--scenario-id", required=True, type=str)
    iter_parser.add_argument("--purpose", required=True, type=str)
    iter_parser.add_argument("--config", type=str, default="qa_agent.yaml")
    iter_parser.add_argument("--runs-dir", type=str, default="runs")
    iter_parser.add_argument("--verbose", "-v", action="store_true")

    # ── onboard-questions ─────────────────────────────────────────────────────
    oq_parser = subparsers.add_parser(
        "onboard-questions",
        help="Generate onboarding questions for an agent (JSON output, no synthesis)",
    )
    oq_parser.add_argument("--scenario-id", required=True, type=str)
    oq_parser.add_argument("--config", type=str, default="qa_agent.yaml")
    oq_parser.add_argument("--runs-dir", type=str, default="runs")
    oq_parser.add_argument("--verbose", "-v", action="store_true")

    # ── onboard-synthesize ────────────────────────────────────────────────────
    os_parser = subparsers.add_parser(
        "onboard-synthesize",
        help="Synthesise agent context from collected Q&A pairs (JSON string)",
    )
    os_parser.add_argument("--scenario-id", required=True, type=str)
    os_parser.add_argument(
        "--qa", required=True, type=str,
        help='JSON array of {"question": "...", "answer": "..."} objects',
    )
    os_parser.add_argument("--config", type=str, default="qa_agent.yaml")
    os_parser.add_argument("--runs-dir", type=str, default="runs")
    os_parser.add_argument("--verbose", "-v", action="store_true")

    # ── history ───────────────────────────────────────────────────────────────
    hist_parser = subparsers.add_parser(
        "history",
        help="Show iteration history for a scenario",
    )
    hist_parser.add_argument("--scenario-id", required=True, type=str)
    hist_parser.add_argument("--runs-dir", type=str, default="runs")

    # ── serve / dashboard ─────────────────────────────────────────────────────
    serve_parser = subparsers.add_parser("serve", help="Start the run history dashboard (no browser)")
    serve_parser.add_argument("--port", type=int, default=8080)
    serve_parser.add_argument("--runs-dir", type=str, default="runs")

    dash_parser = subparsers.add_parser("dashboard", help="Open the run history dashboard in your browser")
    dash_parser.add_argument("--port", type=int, default=8080)
    dash_parser.add_argument("--runs-dir", type=str, default="runs")

    args = parser.parse_args()

    if args.command == "onboard-questions":
        sys.exit(_onboard_questions(
            scenario_id=args.scenario_id,
            config_path=args.config,
            runs_dir=args.runs_dir,
            verbose=args.verbose,
        ))

    if args.command == "onboard-synthesize":
        sys.exit(_onboard_synthesize(
            scenario_id=args.scenario_id,
            qa_json=args.qa,
            config_path=args.config,
            runs_dir=args.runs_dir,
            verbose=args.verbose,
        ))

    if args.command == "onboard":
        _configure_logging(getattr(args, "verbose", False))
        sys.exit(_onboard(
            scenario_id=args.scenario_id,
            config_path=args.config,
            runs_dir=args.runs_dir,
            verbose=args.verbose,
            auto=args.auto,
        ))

    if args.command == "serve":
        _configure_logging(False)
        serve(port=args.port, runs_dir=args.runs_dir)
        sys.exit(0)

    if args.command == "dashboard":
        _configure_logging(False)
        port = args.port
        _step(f"Starting dashboard at {_c(_CYAN + _BOLD, f'http://localhost:{port}')} …")
        server_thread = threading.Thread(
            target=serve, kwargs={"port": port, "runs_dir": args.runs_dir, "quiet": True}, daemon=True
        )
        server_thread.start()
        time.sleep(0.4)
        webbrowser.open(f"http://localhost:{port}")
        _ok("Dashboard open. Press Ctrl+C to stop.")
        try:
            server_thread.join()
        except KeyboardInterrupt:
            print(f"\n  {_c(_DIM, 'Dashboard stopped.')}")
        sys.exit(0)

    if args.command == "history":
        sys.exit(_history(scenario_id=args.scenario_id, runs_dir=args.runs_dir))

    if args.command == "iterate":
        _configure_logging(args.verbose)
        sys.exit(_run(
            scenario_id=args.scenario_id,
            purpose=args.purpose,
            config_path=args.config,
            runs_dir=args.runs_dir,
            verbose=args.verbose,
            no_browser=True,
        ))

    # ── run or interactive ────────────────────────────────────────────────────
    interactive = args.command is None or not (
        getattr(args, "scenario_id", None) and getattr(args, "purpose", None)
    )

    if interactive:
        scenario_id, purpose = _interactive_prompt()
        config_path = getattr(args, "config", "qa_agent.yaml")
        runs_dir    = getattr(args, "runs_dir", "runs")
        verbose     = getattr(args, "verbose", False)
        no_browser  = False
    else:
        scenario_id = args.scenario_id
        purpose     = args.purpose
        config_path = args.config
        runs_dir    = args.runs_dir
        verbose     = args.verbose
        no_browser  = args.no_browser

    sys.exit(_run(
        scenario_id=scenario_id,
        purpose=purpose,
        config_path=config_path,
        runs_dir=runs_dir,
        verbose=verbose,
        no_browser=no_browser,
    ))


if __name__ == "__main__":
    main()
