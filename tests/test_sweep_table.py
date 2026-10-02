"""Tests for scripts/wakekv_sweep_table.py, the rerank-sweep table.

We cover the pure-Python parts (label extraction, quality-shape parsing,
end-to-end table rendering) so an accidental format drift in the
LongBench JSON doesn't silently drop columns from the paper's Table 4.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys

import pytest

_SCRIPTS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "wakekv_sweep_table", os.path.join(_SCRIPTS, "wakekv_sweep_table.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


tab = _load_module()


# ----------------------------------------------------- label + interval parsing
def test_label_strips_throughput_prefix():
    assert tab._label("/x/y/wakekv-stock.json") == "stock"
    assert tab._label("wakekv-reactive-r4.json") == "reactive-r4"


def test_label_strips_longbench_prefix():
    assert tab._label("wakekv-longbench-reactive-r1.json") == "reactive-r1"


def test_interval_of_reactive_label():
    assert tab._interval_of("reactive-r1") == 1
    assert tab._interval_of("reactive-r16") == 16


def test_interval_of_stock_is_sentinel():
    assert tab._interval_of("stock") == -1


# --------------------------------------------------------- quality-shape parsing
def test_quality_summary_per_task_dict_form():
    data = {"per_task": {"qasper": {"f1": 40.1}, "triviaqa": {"f1": 85.0}}}
    mean, per = tab._quality_summary(data)
    assert per == {"qasper": 40.1, "triviaqa": 85.0}
    assert mean == pytest.approx((40.1 + 85.0) / 2)


def test_quality_summary_top_level_scalars():
    """Older LongBench harnesses just dump {task: score} at top level."""
    data = {"qasper": 40.1, "2wikimqa": 17.3, "model": "mistral", "elapsed_time": 42.0}
    mean, per = tab._quality_summary(data)
    assert set(per) == {"qasper", "2wikimqa"}
    assert per["qasper"] == 40.1


def test_quality_summary_empty_gives_none_mean():
    mean, per = tab._quality_summary({"config": {"budget": 64}})
    assert per == {}
    assert mean is None


# ----------------------------------------------- end-to-end table (throughput only)
def _write(dir_: str, name: str, obj: dict) -> None:
    with open(os.path.join(dir_, name), "w") as f:
        json.dump(obj, f)


def test_table_throughput_only(tmp_path, capsys):
    tp = tmp_path / "tp"
    tp.mkdir()
    _write(str(tp), "wakekv-stock.json",
           {"output_tokens_per_second": 100.0, "elapsed_time": 60.0})
    _write(str(tp), "wakekv-reactive-r1.json",
           {"output_tokens_per_second": 40.0, "elapsed_time": 150.0})
    _write(str(tp), "wakekv-reactive-r16.json",
           {"output_tokens_per_second": 120.0, "elapsed_time": 50.0})

    rc = tab.main(["wakekv_sweep_table.py", str(tp)])
    assert rc == 0
    out = capsys.readouterr().out

    # Stock row is baseline, reactive-r1 is slower, r16 is faster.
    assert "1.00× (baseline)" in out
    assert "0.40×" in out    # r1: 40 / 100
    assert "1.20×" in out    # r16: 120 / 100
    # Rows appear in interval order (r1 before r16), and stock is first.
    stock_pos = out.index("stock")
    r1_pos = out.index("reactive-r1")
    r16_pos = out.index("reactive-r16")
    assert stock_pos < r1_pos < r16_pos


# --------------------------------------------------- end-to-end table (+quality)
def test_table_with_quality_dir(tmp_path, capsys):
    tp = tmp_path / "tp"
    tp.mkdir()
    q = tmp_path / "q"
    q.mkdir()

    _write(str(tp), "wakekv-stock.json",
           {"output_tokens_per_second": 100.0, "elapsed_time": 60.0})
    _write(str(tp), "wakekv-reactive-r1.json",
           {"output_tokens_per_second": 40.0, "elapsed_time": 150.0})

    _write(str(q), "wakekv-longbench-stock.json",
           {"qasper": 40.0, "triviaqa": 85.0})
    _write(str(q), "wakekv-longbench-reactive-r1.json",
           {"qasper": 39.5, "triviaqa": 85.2})

    rc = tab.main(["wakekv_sweep_table.py", str(tp), "--quality-dir", str(q)])
    assert rc == 0
    out = capsys.readouterr().out

    assert "LongBench mean" in out
    # Stock mean = 62.50, reactive mean = 62.35 — both must appear.
    assert "62.50" in out
    assert "62.35" in out
    # Per-task section renders with the union of tasks.
    assert "Per-task LongBench" in out
    assert "qasper" in out and "triviaqa" in out


def test_table_quality_dir_missing_files_shows_dashes(tmp_path, capsys):
    tp = tmp_path / "tp"
    tp.mkdir()
    q = tmp_path / "q"
    q.mkdir()

    _write(str(tp), "wakekv-stock.json",
           {"output_tokens_per_second": 100.0, "elapsed_time": 60.0})
    _write(str(tp), "wakekv-reactive-r1.json",
           {"output_tokens_per_second": 40.0, "elapsed_time": 150.0})
    # Quality only for stock, not reactive — the reactive row should show a dash.
    _write(str(q), "wakekv-longbench-stock.json", {"qasper": 40.0})

    rc = tab.main(["wakekv_sweep_table.py", str(tp), "--quality-dir", str(q)])
    assert rc == 0
    out = capsys.readouterr().out

    # Table should not crash on the missing reactive quality — a dash appears
    # in that row instead. (Look for "reactive-r1 |" followed by the dash.)
    r1_line = [ln for ln in out.splitlines() if "reactive-r1 |" in ln][0]
    assert "— |" in r1_line


def test_missing_throughput_dir_returns_1(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = tab.main(["wakekv_sweep_table.py", str(empty)])
    assert rc == 1
