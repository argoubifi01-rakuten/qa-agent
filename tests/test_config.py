import os
import pytest
from pathlib import Path
from qa_agent.config import load_config, Config


MINIMAL_YAML = """\
qa_llm:
  provider: anthropic
  model: claude-opus-5
target:
  websocket_url: wss://example.com/ws
  auth_url: https://example.com/api/v2/auth/anonymous
  thread_creation_url: https://example.com/api/v2/threads
  wss_response_timeout: 30
test_generation:
  num_test_cases: 5
  max_turns: 1
"""


@pytest.fixture
def yaml_file(tmp_path):
    f = tmp_path / "qa_agent.yaml"
    f.write_text(MINIMAL_YAML)
    return str(f)


def test_load_config_success(yaml_file, monkeypatch):
    monkeypatch.setenv("QA_LLM_API_KEY", "test-api-key")
    monkeypatch.setenv("TARGET_AUTH_SECRET", "test-secret")
    cfg = load_config(yaml_file)
    assert isinstance(cfg, Config)
    assert cfg.qa_llm.model == "claude-opus-5"
    assert cfg.target.wss_response_timeout == 30
    assert cfg.test_generation.num_test_cases == 5
    assert cfg.test_generation.max_turns == 1
    assert cfg.qa_llm_api_key == "test-api-key"
    assert cfg.target_auth_secret == "test-secret"


def test_load_config_missing_api_key(yaml_file, monkeypatch):
    monkeypatch.delenv("QA_LLM_API_KEY", raising=False)
    monkeypatch.setenv("TARGET_AUTH_SECRET", "test-secret")
    with pytest.raises(EnvironmentError, match="QA_LLM_API_KEY"):
        load_config(yaml_file)


def test_load_config_missing_auth_secret(yaml_file, monkeypatch):
    monkeypatch.setenv("QA_LLM_API_KEY", "test-api-key")
    monkeypatch.delenv("TARGET_AUTH_SECRET", raising=False)
    with pytest.raises(EnvironmentError, match="TARGET_AUTH_SECRET"):
        load_config(yaml_file)


def test_load_config_missing_yaml():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/qa_agent.yaml")


def test_load_config_defaults_wss_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_LLM_API_KEY", "k")
    monkeypatch.setenv("TARGET_AUTH_SECRET", "s")
    yaml_no_timeout = """\
qa_llm:
  provider: anthropic
  model: claude-opus-5
target:
  websocket_url: wss://x.com/ws
  auth_url: https://x.com/auth
  thread_creation_url: https://x.com/threads
test_generation:
  num_test_cases: 3
  max_turns: 2
"""
    f = tmp_path / "qa_agent.yaml"
    f.write_text(yaml_no_timeout)
    cfg = load_config(str(f))
    assert cfg.target.wss_response_timeout == 120
