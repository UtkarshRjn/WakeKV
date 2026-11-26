"""Resume / grid helpers for masked-accuracy (no torch dependency)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


def cell_key(task_id: str, mode: str, budget: int | None) -> str:
    return f"{task_id}|{mode}|{budget}"


def load_checkpoint(path: Path) -> dict:
    if not path.exists():
        return {"rows": [], "done": {}}
    data = json.loads(path.read_text())
    done = {
        cell_key(r["task"], r["mode"], r.get("budget")): r for r in data.get("rows", [])
    }
    return {"rows": data.get("rows", []), "done": done, "meta": data}


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, delete=False, suffix=".tmp"
    ) as tmp:
        json.dump(payload, tmp, indent=2)
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def job_grid(budgets: list[int]) -> list[tuple[str, int | None]]:
    jobs: list[tuple[str, int | None]] = [("full", None)]
    for budget in budgets:
        for mode in ("frozen", "evict", "reactive"):
            jobs.append((mode, budget))
    return jobs
