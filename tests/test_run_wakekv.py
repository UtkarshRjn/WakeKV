"""Tests for scripts/run_wakekv.py and wakekv._shimmed_main.

These cover the argv-rewriting that routes the target through the in-process
bootstrap (the fix for the execvp-drops-the-shim bug). The real end-to-end
launch needs a FlexiCache/vLLM process; here we test the
pure logic without spawning vLLM.
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def _load_run_wakekv():
    """Import scripts/run_wakekv.py as a module (it's a script, not a package)."""
    path = os.path.join(_SCRIPTS, "run_wakekv.py")
    spec = importlib.util.spec_from_file_location("run_wakekv", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


run_wakekv = _load_run_wakekv()


# ---------------------------------------------------- build_child_command
def test_rewrites_python_script_target():
    out = run_wakekv.build_child_command(["python", "bench.py", "--n", "8"])
    assert out == ["python", "-m", "wakekv._shimmed_main", "bench.py", "--n", "8"]


def test_rewrites_python_dash_m_target():
    out = run_wakekv.build_child_command(
        ["python", "-m", "vllm.entrypoints.openai.api_server", "--model", "x"]
    )
    assert out == [
        "python", "-m", "wakekv._shimmed_main",
        "-m", "vllm.entrypoints.openai.api_server", "--model", "x",
    ]


def test_strips_leading_double_dash():
    out = run_wakekv.build_child_command(["--", "python", "bench.py"])
    assert out == ["python", "-m", "wakekv._shimmed_main", "bench.py"]


def test_python3_interpreter_accepted():
    out = run_wakekv.build_child_command(["python3.12", "bench.py"])
    assert out[:3] == ["python3.12", "-m", "wakekv._shimmed_main"]


def test_empty_target_returns_empty():
    assert run_wakekv.build_child_command([]) == []
    assert run_wakekv.build_child_command(["--"]) == []


def test_non_python_target_rejected():
    with pytest.raises(ValueError, match="Python interpreter"):
        run_wakekv.build_child_command(["vllm", "serve", "mymodel"])


# ---------------------------------------------------- _shimmed_main dispatch
def test_shimmed_main_off_mode_skips_install_and_runs_target(monkeypatch, tmp_path, capsys):
    """With mode=off, the bootstrap must not attempt to import the shim, and
    must still run the target script."""
    from wakekv import _shimmed_main

    target = tmp_path / "hello.py"
    target.write_text("import sys; print('TARGET-RAN', sys.argv[1])\n")

    monkeypatch.setenv("WAKEKV_MODE", "off")
    monkeypatch.setenv("WAKEKV_RERANK_INTERVAL", "1")
    monkeypatch.setattr(sys, "argv", ["_shimmed_main", str(target), "ARG1"])

    rc = _shimmed_main.main()
    assert rc == 0
    assert "TARGET-RAN ARG1" in capsys.readouterr().out


def test_shimmed_main_installs_shim_when_mode_set(monkeypatch, tmp_path):
    """With a real mode, the bootstrap installs the shim before running the
    target. We stub flexicache_shim.install to avoid needing FlexiCache."""
    from wakekv import _shimmed_main, flexicache_shim

    calls = {}

    def fake_install(mode, rerank_interval):
        calls["mode"] = mode
        calls["interval"] = rerank_interval

    monkeypatch.setattr(flexicache_shim, "install", fake_install)

    target = tmp_path / "noop.py"
    target.write_text("pass\n")

    monkeypatch.setenv("WAKEKV_MODE", "reactive")
    monkeypatch.setenv("WAKEKV_RERANK_INTERVAL", "4")
    monkeypatch.setattr(sys, "argv", ["_shimmed_main", str(target)])

    rc = _shimmed_main.main()
    assert rc == 0
    assert calls == {"mode": "reactive", "interval": 4}


def test_shimmed_main_puts_script_dir_on_path(monkeypatch, tmp_path, capsys):
    """Regression: a target script must be able to import its siblings, like
    `python script.py` does. runpy.run_path alone does NOT add the script's
    directory to sys.path — the bootstrap must."""
    pkgdir = tmp_path / "bench"
    pkgdir.mkdir()
    (pkgdir / "sibling.py").write_text("VALUE = 'sibling-import-ok'\n")
    target = pkgdir / "main.py"
    target.write_text("from sibling import VALUE; print('GOT', VALUE)\n")

    from wakekv import _shimmed_main

    monkeypatch.setenv("WAKEKV_MODE", "off")
    monkeypatch.setenv("WAKEKV_RERANK_INTERVAL", "1")
    # Invoke with a relative path from a different cwd, as the sweep does.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["_shimmed_main", os.path.join("bench", "main.py")])

    rc = _shimmed_main.main()
    assert rc == 0
    assert "GOT sibling-import-ok" in capsys.readouterr().out


def test_shimmed_main_bad_interval_returns_error(monkeypatch):
    from wakekv import _shimmed_main

    monkeypatch.setenv("WAKEKV_MODE", "off")
    monkeypatch.setenv("WAKEKV_RERANK_INTERVAL", "not-an-int")
    monkeypatch.setattr(sys, "argv", ["_shimmed_main", "whatever.py"])
    assert _shimmed_main.main() == 2
