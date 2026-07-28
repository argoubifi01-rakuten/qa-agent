import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from qa_agent.config import load_config
from qa_agent.generator.generator import generate_test_cases
from qa_agent.runner.websocket_runner import WebSocketRunner
from qa_agent.evaluator.evaluator import evaluate
from qa_agent.reporter.reporter import report


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        prog="qa-agent",
        description="Automatically test an LLM agent with generated test cases.",
    )
    subparsers = parser.add_subparsers(dest="command")
    run_parser = subparsers.add_parser("run", help="Run QA evaluation")
    run_parser.add_argument("--agent-description", type=str, default=None)
    run_parser.add_argument("--prompt-file", type=str, default=None)
    run_parser.add_argument("--env", type=str, default="dev")
    run_parser.add_argument("--output", type=str, default=None)
    run_parser.add_argument("--config", type=str, default="qa_agent.yaml")

    args = parser.parse_args()

    if args.command != "run":
        parser.print_help()
        sys.exit(1)

    if not args.agent_description and not args.prompt_file:
        print(
            "Error: at least one of --agent-description or --prompt-file is required.",
            file=sys.stderr,
        )
        sys.exit(1)

    prompt_text: str | None = None
    if args.prompt_file:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            print(f"Error: prompt file not found: {args.prompt_file}", file=sys.stderr)
            sys.exit(1)
        prompt_text = prompt_path.read_text()

    try:
        config = load_config(args.config)
    except (FileNotFoundError, EnvironmentError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Generating test cases...")
    test_cases = generate_test_cases(
        description=args.agent_description,
        prompt=prompt_text,
        config=config,
    )
    print(f"Generated {len(test_cases)} test cases.")

    print("Running test cases against target agent...")
    runner = WebSocketRunner(
        config=config,
        description=args.agent_description,
        prompt=prompt_text,
    )
    run_results = [runner.run(tc) for tc in test_cases]

    print("Evaluating results...")
    eval_results = evaluate(
        run_results=run_results,
        description=args.agent_description,
        prompt=prompt_text,
        config=config,
    )

    exit_code = report(
        eval_results=eval_results,
        description=args.agent_description,
        config=config,
        output_path=args.output,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
