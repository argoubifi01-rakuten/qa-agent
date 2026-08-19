import os
import pytest
from pathlib import Path
from qa_agent.config import load_config, Config


MINIMAL_YAML = """\
qa_llm:
  provider: openai
  model: gpt-4o
target:
  websocket_url: wss://example.com/ws
  auth_url: https://example.com/api/v2/auth/anonymous
  thread_creation_url: https://example.com/api/v2/threads
  scenario_id: test-scenario
  wss_response_timeout: 30
test_generation:
  num_test_cases: 5
  max_turns: 1
"""

_REQUIRED_ENV = {
    "OPENAI_API_KEY": "test-api-key",
    "EVAL_MOCK_SECRET": "test-mock-secret",
    "MONGODB_URI": "mongodb://localhost/testdb",
}


@pytest.fixture
def yaml_file(tmp_path):
    f = tmp_path / "qa_agent.yaml"
    f.write_text(MINIMAL_YAML)
    return str(f)


def test_load_config_success(yaml_file, monkeypatch):
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    cfg = load_config(yaml_file)
    assert isinstance(cfg, Config)
    assert cfg.qa_llm.model == "gpt-4o"
    assert cfg.target.wss_response_timeout == 30
    assert cfg.target.scenario_id == "test-scenario"
    assert cfg.test_generation.num_test_cases == 5
    assert cfg.test_generation.max_turns == 1
    assert cfg.qa_llm_api_key == "test-api-key"
    assert cfg.eval_mock_secret == "test-mock-secret"
    assert cfg.mongodb_uri == "mongodb://localhost/testdb"


def test_load_config_missing_api_key(yaml_file, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("EVAL_MOCK_SECRET", "s")
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost/db")
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        load_config(yaml_file)


def test_load_config_missing_eval_mock_secret(yaml_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.delenv("EVAL_MOCK_SECRET", raising=False)
    monkeypatch.setenv("MONGODB_URI", "mongodb://localhost/db")
    with pytest.raises(EnvironmentError, match="EVAL_MOCK_SECRET"):
        load_config(yaml_file)


def test_load_config_missing_mongodb_uri(yaml_file, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    monkeypatch.setenv("EVAL_MOCK_SECRET", "s")
    monkeypatch.delenv("MONGODB_URI", raising=False)
    with pytest.raises(EnvironmentError, match="MONGODB_URI"):
        load_config(yaml_file)


def test_load_config_missing_yaml():
    with pytest.raises(FileNotFoundError):
        load_config("/nonexistent/path/qa_agent.yaml")


def test_load_config_defaults_wss_timeout(tmp_path, monkeypatch):
    for k, v in _REQUIRED_ENV.items():
        monkeypatch.setenv(k, v)
    yaml_no_timeout = """\
qa_llm:
  provider: openai
  model: gpt-4o
target:
  websocket_url: wss://x.com/ws
  auth_url: https://x.com/auth
  thread_creation_url: https://x.com/threads
  scenario_id: my-scenario
test_generation:
  num_test_cases: 3
  max_turns: 2
"""
    f = tmp_path / "qa_agent.yaml"
    f.write_text(yaml_no_timeout)
    cfg = load_config(str(f))
    assert cfg.target.wss_response_timeout == 120
