import json
import logging
import os
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

_RUNS_DIR = "runs"


def _scenario_dir(scenario_id: str, runs_dir: str) -> str:
    return os.path.join(runs_dir, scenario_id)


def get_iteration_number(scenario_id: str, runs_dir: str = _RUNS_DIR) -> int:
    """Return the next 1-based iteration number for a scenario."""
    return len(load_scenario_runs(scenario_id, runs_dir)) + 1


def save_run(run_data: dict, runs_dir: str = _RUNS_DIR) -> str:
    """Save run to runs/<scenario_id>/run-<N>-<timestamp>.json. Returns the saved path."""
    scenario_id = run_data.get("scenario_id", "unknown")
    iteration = run_data.get("iteration", 1)
    sdir = _scenario_dir(scenario_id, runs_dir)
    os.makedirs(sdir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    filename = f"run-{iteration:03d}-{ts}.json"
    path = os.path.join(sdir, filename)
    with open(path, "w") as f:
        json.dump(run_data, f, indent=2)
    logger.info("Run saved to %s", path)
    return path


def load_scenario_runs(scenario_id: str, runs_dir: str = _RUNS_DIR) -> list[dict]:
    """Load all runs for a scenario, sorted oldest-first (by filename / iteration)."""
    sdir = _scenario_dir(scenario_id, runs_dir)
    if not os.path.isdir(sdir):
        return []
    runs = []
    for fname in sorted(os.listdir(sdir)):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(sdir, fname)
        try:
            with open(path) as f:
                data = json.load(f)
            data["_filename"] = fname
            runs.append(data)
        except Exception as exc:
            logger.warning("Could not load run file %s: %s", path, exc)
    return runs


def load_runs(runs_dir: str = _RUNS_DIR) -> list[dict]:
    """Load all run JSON files (flat legacy + per-scenario subdirectory), sorted newest-first."""
    if not os.path.isdir(runs_dir):
        return []
    runs = []
    for entry in os.listdir(runs_dir):
        entry_path = os.path.join(runs_dir, entry)
        if os.path.isfile(entry_path) and entry.endswith(".json"):
            try:
                with open(entry_path) as f:
                    data = json.load(f)
                data["_filename"] = entry
                runs.append(data)
            except Exception as exc:
                logger.warning("Could not load run file %s: %s", entry_path, exc)
        elif os.path.isdir(entry_path):
            for fname in os.listdir(entry_path):
                if not fname.endswith(".json"):
                    continue
                path = os.path.join(entry_path, fname)
                try:
                    with open(path) as f:
                        data = json.load(f)
                    data["_filename"] = fname
                    runs.append(data)
                except Exception as exc:
                    logger.warning("Could not load run file %s: %s", path, exc)
    runs.sort(key=lambda r: r.get("timestamp", ""), reverse=True)
    return runs
