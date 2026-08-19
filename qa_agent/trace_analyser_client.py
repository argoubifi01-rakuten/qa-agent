"""
Client for the trace-analyser pull API.

Usage:
    client = TraceAnalyserClient.from_env()  # reads TRACE_ANALYSER_URL
    queries = client.fetch_queries(tool="shopping", score_name="tool_error", limit=10)
    # returns list of {"trace_id": ..., "user_input": ..., "timestamp": ..., "scores": [...], "langfuse_url": ...}

    queries = client.fetch_dataset("shopping-failures-2026-08")
    # returns items from a named dataset
"""
import os
import httpx
from dataclasses import dataclass
from typing import Any


@dataclass
class ProductionQuery:
    trace_id: str | None
    user_input: str
    timestamp: str | None
    scores: list[dict]
    langfuse_url: str | None


class TraceAnalyserClient:
    def __init__(self, base_url: str):
        self._base = base_url.rstrip("/")

    @classmethod
    def from_env(cls) -> "TraceAnalyserClient":
        url = os.environ.get("TRACE_ANALYSER_URL")
        if not url:
            raise RuntimeError("TRACE_ANALYSER_URL is not set")
        return cls(url)

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        clean = {k: v for k, v in params.items() if v is not None}
        resp = httpx.get(f"{self._base}{path}", params=clean, timeout=30)
        resp.raise_for_status()
        return resp.json()

    def fetch_queries(
        self,
        tool: str | None = None,
        score_name: str | None = None,
        score_min: float = 0.0,
        date_from: str | None = None,
        date_to: str | None = None,
        limit: int = 20,
    ) -> list[ProductionQuery]:
        """Fetch real user queries from production filtered by tool/score."""
        data = self._get("/api/traces/export", {
            "tool": tool,
            "score_name": score_name,
            "score_min": score_min,
            "date_from": date_from,
            "date_to": date_to,
            "limit": limit,
        })
        return [ProductionQuery(**q) for q in data.get("queries", [])]

    def fetch_dataset(self, name: str) -> list[ProductionQuery]:
        """Fetch all items from a named trace-analyser dataset."""
        data = self._get("/api/traces/export", {"dataset": name, "limit": 100})
        return [ProductionQuery(**q) for q in data.get("queries", [])]
