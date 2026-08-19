import os
import yaml
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QALLMConfig:
    provider: str
    model: str


@dataclass
class TargetConfig:
    websocket_url: str
    auth_url: str
    thread_creation_url: str
    scenario_id: str
    wss_response_timeout: int = 120


@dataclass
class TestGenerationConfig:
    num_test_cases: int
    max_turns: int


@dataclass
class TraceAnalyserConfig:
    url: str
    dataset: str | None = None
    tool_filter: str | None = None
    score_filter: str | None = None
    limit: int = 20


@dataclass
class Config:
    qa_llm: QALLMConfig
    target: TargetConfig
    test_generation: TestGenerationConfig
    qa_llm_api_key: str
    eval_mock_secret: str
    mongodb_uri: str | None  # None → use Firestore via ADC (no VPN required)
    trace_analyser: TraceAnalyserConfig | None = None


def load_config(yaml_path: str) -> Config:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    missing = []
    api_key = os.environ.get("OPENAI_API_KEY")
    eval_mock_secret = os.environ.get("EVAL_MOCK_SECRET")
    mongodb_uri = os.environ.get("MONGODB_URI")  # optional — falls back to Firestore ADC

    if not api_key:
        missing.append("OPENAI_API_KEY")
    if not eval_mock_secret:
        missing.append("EVAL_MOCK_SECRET")

    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            "See .env.example for required variables."
        )

    target_raw = raw["target"]
    return Config(
        qa_llm=QALLMConfig(
            provider=raw["qa_llm"]["provider"],
            model=raw["qa_llm"]["model"],
        ),
        target=TargetConfig(
            websocket_url=target_raw["websocket_url"],
            auth_url=target_raw["auth_url"],
            thread_creation_url=target_raw["thread_creation_url"],
            scenario_id=target_raw["scenario_id"],
            wss_response_timeout=target_raw.get("wss_response_timeout", 120),
        ),
        test_generation=TestGenerationConfig(
            num_test_cases=raw["test_generation"]["num_test_cases"],
            max_turns=raw["test_generation"]["max_turns"],
        ),
        qa_llm_api_key=api_key,
        eval_mock_secret=eval_mock_secret,
        mongodb_uri=mongodb_uri,
        trace_analyser=_load_trace_analyser_config(raw),
    )


def _load_trace_analyser_config(raw: dict) -> "TraceAnalyserConfig | None":
    ta_raw = raw.get("trace_analyser")
    if not ta_raw:
        url = os.environ.get("TRACE_ANALYSER_URL")
        if not url:
            return None
        ta_raw = {"url": url}
    return TraceAnalyserConfig(
        url=ta_raw.get("url") or os.environ.get("TRACE_ANALYSER_URL", ""),
        dataset=ta_raw.get("dataset"),
        tool_filter=ta_raw.get("tool_filter"),
        score_filter=ta_raw.get("score_filter"),
        limit=ta_raw.get("limit", 20),
    )
