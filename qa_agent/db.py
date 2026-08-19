"""Database access layer.

Supports two backends, selected by whether MONGODB_URI is set:

  Firestore (default) — google-cloud-firestore SDK, ADC auth, no VPN required.
  MongoDB  (legacy)   — PyMongo, MONGODB_URI env var, requires VPN / private network.

All public functions accept ``mongodb_uri: str | None``.  Pass None (or omit) to
use Firestore.  Pass the URI string to use the MongoDB wire-protocol path.
"""

import difflib
import logging

logger = logging.getLogger(__name__)

_DEFAULT_BASELINE_ID = "6812e64f9dfaf301f7000001"


def _is_object_id(s: str) -> bool:
    return len(s) == 24 and all(c in "0123456789abcdefABCDEF" for c in s)


# ---------------------------------------------------------------------------
# Session abstraction
# ---------------------------------------------------------------------------

def _firestore_credentials():
    """Return GCP credentials from GCP_CREDENTIALS_BASE64 env var, or None for ADC.

    GCP_CREDENTIALS_BASE64 should be a base64-encoded service account JSON string.
    If not set, google-cloud-firestore falls back to Application Default Credentials
    (gcloud auth application-default login, GOOGLE_APPLICATION_CREDENTIALS, or
    Workload Identity on GCP).
    """
    import os, base64, json
    b64 = os.environ.get("GCP_CREDENTIALS_BASE64")
    if not b64:
        return None
    from google.oauth2 import service_account  # type: ignore[import]
    info = json.loads(base64.b64decode(b64).decode())
    return service_account.Credentials.from_service_account_info(
        info,
        scopes=["https://www.googleapis.com/auth/datastore"],
    )


def _firestore_database_id() -> str | None:
    """Return Firestore database ID to use, or None to let the SDK default to '(default)'.

    Priority:
    1. GOOGLE_FIRESTORE_DATABASE env var (explicit override)
    2. Database path extracted from MONGODB_URI (same database, native API)
    """
    import os
    explicit = os.environ.get("GOOGLE_FIRESTORE_DATABASE")
    if explicit:
        return explicit
    uri = os.environ.get("MONGODB_URI")
    if uri:
        from urllib.parse import urlparse
        path = urlparse(uri).path.lstrip("/")
        db = path.split("?")[0].split("/")[0]
        if db:
            return db
    return None


class _FirestoreSession:
    """Thin wrapper around google-cloud-firestore for the queries we need."""

    def __init__(self) -> None:
        import os
        from google.cloud import firestore  # type: ignore[import]
        creds = _firestore_credentials()
        project = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCLOUD_PROJECT")
        db_id = _firestore_database_id()
        kwargs: dict = {"credentials": creds}
        if project:
            kwargs["project"] = project
        if db_id:
            kwargs["database"] = db_id
        self._fs = firestore.Client(**kwargs)

    @staticmethod
    def _doc_id(raw_id: str) -> str:
        """Convert a raw ObjectId hex string to the Firestore MongoDB-compat document ID.

        Firestore stores MongoDB ObjectId-based document IDs as ``__oid<hex>__``.
        If the id already has that wrapping or isn't a 24-char hex string, return as-is.
        """
        if raw_id.startswith("__oid") and raw_id.endswith("__"):
            return raw_id
        if len(raw_id) == 24 and all(c in "0123456789abcdefABCDEF" for c in raw_id):
            return f"__oid{raw_id}__"
        return raw_id

    def find_one_by_id(self, collection: str, doc_id: str) -> dict | None:
        doc = self._fs.collection(collection).document(self._doc_id(doc_id)).get()
        return doc.to_dict() if doc.exists else None

    def find_one_by_field(self, collection: str, field: str, value: object) -> dict | None:
        docs = list(
            self._fs.collection(collection)
            .where(field, "==", value)
            .limit(1)
            .stream()
        )
        return docs[0].to_dict() if docs else None

    def find_by_field(self, collection: str, field: str, value: object) -> list[dict]:
        """Return all docs where field == value, ordered by document name."""
        docs = (
            self._fs.collection(collection)
            .where(field, "==", value)
            .order_by("__name__")
            .stream()
        )
        return [d.to_dict() for d in docs]

    def find_by_field_in(self, collection: str, field: str, values: list) -> list[dict]:
        """Return docs where field is in values (batched to respect Firestore 30-item limit)."""
        results: list[dict] = []
        for i in range(0, len(values), 30):
            batch = values[i : i + 30]
            docs = self._fs.collection(collection).where(field, "in", batch).stream()
            results.extend(d.to_dict() for d in docs)
        return results

    def close(self) -> None:
        pass


class _MongoSession:
    """Thin wrapper around PyMongo for the queries we need."""

    def __init__(self, mongodb_uri: str) -> None:
        from pymongo import MongoClient  # type: ignore[import]
        from bson import ObjectId  # type: ignore[import]
        self._client = MongoClient(mongodb_uri)
        self._db = self._client.get_default_database()
        self._ObjectId = ObjectId

    def _oid(self, s: str):
        return self._ObjectId(s)

    def find_one_by_id(self, collection: str, doc_id: str) -> dict | None:
        return self._db[collection].find_one({"_id": self._oid(doc_id)})

    def find_one_by_field(self, collection: str, field: str, value: object) -> dict | None:
        return self._db[collection].find_one({field: value})

    def find_by_field(self, collection: str, field: str, value: object) -> list[dict]:
        return list(self._db[collection].find({field: value}).sort("_id", 1))

    def find_by_field_in(self, collection: str, field: str, values: list) -> list[dict]:
        return list(self._db[collection].find({field: {"$in": values}}))

    def close(self) -> None:
        self._client.close()


def _make_session(mongodb_uri: str | None):
    if mongodb_uri:
        return _MongoSession(mongodb_uri)
    return _FirestoreSession()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_scenario_prompt(
    session, scenario_id: str
) -> tuple[str | None, str | None, str | None]:
    """Return (agent_name, task_agent_id, prompt) for a scenario ID."""
    doc = None
    if _is_object_id(scenario_id):
        doc = session.find_one_by_id("scenario_agents", scenario_id)
        if doc is None:
            doc = session.find_one_by_id("scenario_agent_snapshots", scenario_id)
    if doc is None:
        doc = session.find_one_by_field("scenario_agents", "scenarioAgentId", scenario_id)
    if doc is None:
        doc = session.find_one_by_field("scenario_agent_snapshots", "scenarioAgentId", scenario_id)
    if doc is None:
        logger.warning("No scenario document found for scenario_id=%s", scenario_id)
        return None, None, None

    scenario_name = doc.get("name") or doc.get("scenarioAgentId", scenario_id)
    eg = doc.get("executionGraph", {})
    start_id = eg.get("startTaskAgentId")
    if not start_id:
        logger.warning("No startTaskAgentId in executionGraph for scenario_id=%s", scenario_id)
        return scenario_name, None, None

    ta = session.find_one_by_id("task_agents", start_id)
    if not ta:
        logger.warning("task_agent %s not found for scenario_id=%s", start_id, scenario_id)
        return scenario_name, start_id, None

    instr = ta.get("instructions", {})
    prompt = instr.get("prompt", "") if isinstance(instr, dict) else ""
    agent_name = f"{scenario_name} / {ta.get('name', start_id)}"
    logger.debug(
        "Resolved prompt for %s via task_agent %s (%d chars)",
        scenario_id, start_id, len(prompt),
    )
    return agent_name, start_id, prompt or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_system_prompt(scenario_id: str, mongodb_uri: str | None = None) -> str | None:
    """Fetch the main agent prompt for a scenario. Returns prompt string or None."""
    logger.debug("Fetching system prompt for scenario_id=%s", scenario_id)
    session = _make_session(mongodb_uri)
    try:
        _, _, prompt = _resolve_scenario_prompt(session, scenario_id)
        if prompt is None:
            logger.warning("No prompt found for scenario_id=%s", scenario_id)
        return prompt
    finally:
        session.close()


def fetch_agent_data(scenario_id: str, mongodb_uri: str | None = None) -> dict:
    """Fetch agent name, system prompt, and tool descriptions in one DB session.

    Returns a dict with keys: name, prompt, tools (list of {name, description}).
    """
    logger.debug("Fetching agent data for scenario_id=%s", scenario_id)
    session = _make_session(mongodb_uri)
    try:
        name, task_agent_id, prompt = _resolve_scenario_prompt(session, scenario_id)

        tools: list[dict] = []
        if task_agent_id:
            ta = session.find_one_by_id("task_agents", task_agent_id)
            if ta:
                tool_names = ta.get("toolNames") or []
                if tool_names:
                    desc_map = {
                        d["name"]: d.get("description", "")
                        for d in session.find_by_field_in("tools", "name", tool_names)
                    }
                    for tool_name in tool_names:
                        tools.append({
                            "name": tool_name,
                            "description": desc_map.get(tool_name, ""),
                        })
                else:
                    for t in ta.get("tools") or []:
                        tool_name = t.get("name", "")
                        if tool_name:
                            tools.append({
                                "name": tool_name,
                                "description": t.get("description", ""),
                            })

        logger.debug(
            "Agent data fetched: name=%r prompt=%d chars tools=%d",
            name, len(prompt or ""), len(tools),
        )
        return {"name": name or scenario_id, "prompt": prompt or "", "tools": tools}
    finally:
        session.close()


def fetch_llm_interaction_docs(
    thread_id: str, mongodb_uri: str | None = None
) -> list[dict]:
    """Return all llm_interaction documents for a thread, ordered by creation time."""
    logger.debug("Fetching llm_interaction docs for thread_id=%s", thread_id)
    session = _make_session(mongodb_uri)
    try:
        return session.find_by_field("llm_interactions", "threadId", thread_id)
    finally:
        session.close()


def fetch_prompt_comparison(
    scenario_id: str,
    baseline_id: str,
    mongodb_uri: str | None = None,
) -> dict:
    """Fetch and diff prompts for two scenarios."""
    logger.info("Fetching prompt comparison: %s vs %s", scenario_id, baseline_id)
    session = _make_session(mongodb_uri)
    try:
        tested_name, _, tested_prompt = _resolve_scenario_prompt(session, scenario_id)
        baseline_name, _, baseline_prompt = _resolve_scenario_prompt(session, baseline_id)
    finally:
        session.close()

    a_lines = (baseline_prompt or "").splitlines(keepends=True)
    b_lines = (tested_prompt or "").splitlines(keepends=True)
    diff = list(difflib.unified_diff(
        a_lines, b_lines,
        fromfile=f"baseline ({baseline_name or baseline_id})",
        tofile=f"tested ({tested_name or scenario_id})",
        lineterm="",
    ))

    added   = sum(1 for line in diff if line.startswith("+") and not line.startswith("+++"))
    removed = sum(1 for line in diff if line.startswith("-") and not line.startswith("---"))
    logger.info("Prompt diff: +%d -%d lines", added, removed)

    return {
        "tested_id": scenario_id,
        "tested_name": tested_name,
        "tested_prompt": tested_prompt,
        "baseline_id": baseline_id,
        "baseline_name": baseline_name,
        "baseline_prompt": baseline_prompt,
        "diff_lines": diff,
        "diff_added": added,
        "diff_removed": removed,
    }
