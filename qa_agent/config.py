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
    wss_response_timeout: int = 120


@dataclass
class TestGenerationConfig:
    num_test_cases: int
    max_turns: int


@dataclass
class Config:
    qa_llm: QALLMConfig
    target: TargetConfig
    test_generation: TestGenerationConfig
    qa_llm_api_key: str
    target_auth_secret: str


def load_config(yaml_path: str) -> Config:
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    missing = []
    api_key = os.environ.get("QA_LLM_API_KEY")
    auth_secret = os.environ.get("TARGET_AUTH_SECRET")

    if not api_key:
        missing.append("QA_LLM_API_KEY")
    if not auth_secret:
        missing.append("TARGET_AUTH_SECRET")

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
            wss_response_timeout=target_raw.get("wss_response_timeout", 120),
        ),
        test_generation=TestGenerationConfig(
            num_test_cases=raw["test_generation"]["num_test_cases"],
            max_turns=raw["test_generation"]["max_turns"],
        ),
        qa_llm_api_key=api_key,
        target_auth_secret=auth_secret,
    )
