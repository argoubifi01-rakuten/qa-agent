from unittest.mock import MagicMock, patch
from bson import ObjectId
from qa_agent.db import fetch_system_prompt


_TASK_AGENT_ID = "685e595554dccaf796858642"
_SCENARIO_ID   = "6812e64f9dfaf301f7000001"


def _make_mongo_mock(scenario_doc, task_agent_doc=None):
    """Return a MongoClient mock where each collection is keyed by name."""
    def make_collection(doc):
        col = MagicMock()
        col.find_one.return_value = doc
        return col

    scenario_col   = make_collection(scenario_doc)
    snapshot_col   = make_collection(None)
    task_agent_col = make_collection(task_agent_doc)

    def getitem(name):
        if name == "scenario_agents":
            return scenario_col
        if name == "scenario_agent_snapshots":
            return snapshot_col
        if name == "task_agents":
            return task_agent_col
        return make_collection(None)

    mock_db = MagicMock()
    mock_db.__getitem__ = MagicMock(side_effect=getitem)

    mock_client = MagicMock()
    mock_client.get_default_database.return_value = mock_db
    mock_client.close = MagicMock()
    return mock_client


def test_fetch_system_prompt_returns_prompt():
    scenario_doc = {
        "_id": ObjectId(_SCENARIO_ID),
        "name": "Main Agent",
        "executionGraph": {"startTaskAgentId": _TASK_AGENT_ID},
    }
    task_agent_doc = {
        "_id": ObjectId(_TASK_AGENT_ID),
        "name": "Ninja(5.4)",
        "instructions": {"prompt": "You are a helpful assistant."},
    }
    mock_client = _make_mongo_mock(scenario_doc, task_agent_doc)
    with patch("qa_agent.db.MongoClient", return_value=mock_client):
        result = fetch_system_prompt(_SCENARIO_ID, "mongodb://localhost/testdb")
    assert result == "You are a helpful assistant."
    mock_client.close.assert_called_once()


def test_fetch_system_prompt_returns_none_when_no_doc():
    mock_client = _make_mongo_mock(None, None)
    with patch("qa_agent.db.MongoClient", return_value=mock_client):
        result = fetch_system_prompt("aabbccddeeff001122334455", "mongodb://localhost/testdb")
    assert result is None


def test_fetch_system_prompt_returns_none_when_no_prompt_in_task_agent():
    scenario_doc = {
        "_id": ObjectId(_SCENARIO_ID),
        "executionGraph": {"startTaskAgentId": _TASK_AGENT_ID},
    }
    task_agent_doc = {
        "_id": ObjectId(_TASK_AGENT_ID),
        "instructions": {},
    }
    mock_client = _make_mongo_mock(scenario_doc, task_agent_doc)
    with patch("qa_agent.db.MongoClient", return_value=mock_client):
        result = fetch_system_prompt(_SCENARIO_ID, "mongodb://localhost/testdb")
    assert result is None


def test_fetch_system_prompt_returns_none_when_no_start_task_agent_id():
    scenario_doc = {
        "_id": ObjectId(_SCENARIO_ID),
        "executionGraph": {},
    }
    mock_client = _make_mongo_mock(scenario_doc, None)
    with patch("qa_agent.db.MongoClient", return_value=mock_client):
        result = fetch_system_prompt(_SCENARIO_ID, "mongodb://localhost/testdb")
    assert result is None
