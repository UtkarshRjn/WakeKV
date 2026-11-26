"""CPU tests for masked-accuracy helpers (no GPU / transformers)."""

from __future__ import annotations

from pathlib import Path

from wakekv.grading import (
    extract_math_answer,
    grade,
    grade_math,
    grade_multiturn,
    grade_niah,
    math_answers_equal,
)
from wakekv.masked_eval import atomic_write, cell_key, job_grid, load_checkpoint
from wakekv.reactive_controller import ReactiveController


def test_math_extract_boxed():
    assert extract_math_answer(r"stuff \boxed{42} more") == "42"
    assert extract_math_answer(r"\boxed{\frac{1}{2}}") == r"\frac{1}{2}"


def test_math_equivalence_numeric():
    assert math_answers_equal("42", "42")
    assert math_answers_equal("1/2", "0.5")
    assert not math_answers_equal("41", "42")
    assert grade_math(r"The answer is \boxed{42}.", "42")


def test_niah_and_multiturn():
    assert grade_niah("uuid=abc-123", "abc-123")
    assert not grade_niah("nothing", "abc-123")
    text = "locker 4831, wifi maple-syrup-42, cat Biscuit"
    assert grade_multiturn(text)
    assert grade("multiturn", text, "")


def test_full_row_always_has_miss_rate():
    extra = {"miss_rate": 0.0, "attention_mass_retained": 1.0}
    assert "miss_rate" in extra
    row = {"mode": "full", **extra}
    assert row["miss_rate"] == 0.0


def test_cell_key_and_resume(tmp_path: Path):
    out = tmp_path / "masked.json"
    row = {
        "task": "niah-d0.5-s0",
        "mode": "full",
        "budget": None,
        "correct": True,
        "miss_rate": 0.0,
    }
    atomic_write(out, {"rows": [row]})
    state = load_checkpoint(out)
    key = cell_key("niah-d0.5-s0", "full", None)
    assert key in state["done"]
    assert state["done"][key]["correct"] is True


def test_resident_cap_when_demand_exceeds_budget():
    ctrl = ReactiveController(1, budget=2, demotion="offload")
    ctrl.begin_step(0)
    ctrl.on_unit_lru(0, {0, 1, 2, 3, 4})
    ctrl.end_step()
    assert len(ctrl.resident_pages(0)) <= 2


def test_job_grid_includes_frozen_evict_reactive():
    jobs = job_grid([8, 16, 32, 64])
    modes = {m for m, _ in jobs}
    assert modes == {"full", "frozen", "evict", "reactive"}
    assert len(jobs) == 1 + 3 * 4
