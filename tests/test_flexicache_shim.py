"""Tests for the FlexiCache shim.

These tests do not import FlexiCache. They call ``_apply_config_override``
on a mock config. That is the whole policy change: clear ``unstable_heads``
and set the rerank interval. Running it inside vLLM is a separate machine.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from wakekv import flexicache_shim as shim


@pytest.fixture(autouse=True)
def _reset_shim_state():
    """Every test starts with a fresh module state."""
    shim._original_build = None
    shim._current_mode = "off"
    shim._current_rerank_interval = 16
    shim._last_report = None
    yield
    shim._original_build = None
    shim._current_mode = "off"
    shim._last_report = None


def _mock_config(unstable_heads: list[Any], rerank_frequency: int) -> Any:
    """Look like a FlexiCacheConfig for override purposes."""
    return SimpleNamespace(
        unstable_heads=list(unstable_heads),
        rerank_frequency=rerank_frequency,
        num_unstable_heads=len(unstable_heads),
        unstable_heads_portion=len(unstable_heads) / 256.0,
    )


# --------------------------------------------------------- input validation
def test_install_rejects_unknown_mode():
    with pytest.raises(ValueError, match="mode must be"):
        shim.install(mode="hopeful")


def test_install_rejects_bad_interval():
    with pytest.raises(ValueError, match="rerank_interval"):
        shim.install(mode="reactive", rerank_interval=0)


# -------------------------------------------------- override in reactive mode
def test_reactive_forces_no_unstable_heads():
    shim._current_mode = "reactive"
    shim._current_rerank_interval = 1
    cfg = _mock_config(unstable_heads=[[3, 5], [4, 7]], rerank_frequency=16)
    report = shim._apply_config_override(cfg)
    assert cfg.unstable_heads == []
    assert cfg.rerank_frequency == 1
    assert report.unstable_heads_count_before == 2
    assert report.unstable_heads_count_after == 0
    assert report.rerank_frequency_before == 16
    assert report.rerank_frequency_after == 1


def test_reactive_zeros_scalar_unstable_head_fields():
    """Regression: reactive must zero the scalar num_unstable_heads (and
    unstable_heads_portion), not just the list. Leaving num_unstable_heads>0
    while unstable_heads==[] sends FlexiCache's GPU block-count assertion down
    the wrong branch (off-by-one crash at engine init)."""
    shim._current_mode = "reactive"
    shim._current_rerank_interval = 1
    cfg = _mock_config(unstable_heads=[[3, 5], [4, 7]], rerank_frequency=16)
    shim._apply_config_override(cfg)
    assert cfg.num_unstable_heads == 0
    assert cfg.unstable_heads_portion == 0.0


def test_reactive_respects_configured_interval():
    shim._current_mode = "reactive"
    shim._current_rerank_interval = 4
    cfg = _mock_config(unstable_heads=[[1, 2]], rerank_frequency=16)
    shim._apply_config_override(cfg)
    assert cfg.rerank_frequency == 4


# ------------------------------------------------------ identity mode passthrough
def test_identity_mode_leaves_config_untouched():
    shim._current_mode = "identity"
    original = [[10, 3], [12, 7], [15, 0]]
    cfg = _mock_config(unstable_heads=original, rerank_frequency=16)
    report = shim._apply_config_override(cfg)
    assert cfg.unstable_heads == original
    assert cfg.rerank_frequency == 16
    assert report.mode == "identity"
    assert report.unstable_heads_count_before == report.unstable_heads_count_after


# ------------------------------------------------------------------ reports
def test_report_captures_before_after():
    shim._current_mode = "reactive"
    shim._current_rerank_interval = 2
    cfg = _mock_config(unstable_heads=[[1, 1], [2, 2]], rerank_frequency=16)
    r = shim._apply_config_override(cfg)
    text = str(r)
    assert "reactive" in text
    assert "16" in text and "→ 2" in text
    assert shim.last_report() is r


def test_bad_mode_raised_at_override_time():
    shim._current_mode = "no-such-mode"
    cfg = _mock_config(unstable_heads=[], rerank_frequency=16)
    with pytest.raises(RuntimeError, match="unexpected shim mode"):
        shim._apply_config_override(cfg)


# ---------------------------------------------- shim doesn't touch real FlexiCache
def test_install_errors_cleanly_without_flexicache():
    """We're running in the tests venv, which has no vllm/flexicache. The
    shim must raise a clear message, not import-crash."""
    with pytest.raises(RuntimeError, match="Cannot import"):
        shim.install(mode="reactive", rerank_interval=1)


def test_uninstall_is_a_noop_when_not_installed():
    # Should not raise even though FlexiCache isn't importable.
    shim.uninstall()  # no exception


# ------------------------------------------------- integration with a fake module
def test_install_patches_a_fake_config_class(monkeypatch):
    """Simulate a FlexiCacheConfig class in an importable module, verify
    install() correctly wraps _build."""
    build_calls = []

    class FakeConfig:
        def _build(self):
            build_calls.append(("original", self.unstable_heads, self.rerank_frequency))

    fake_module = SimpleNamespace(FlexiCacheConfig=FakeConfig)

    import sys
    monkeypatch.setitem(sys.modules, "vllm.v1.flexicache.config", fake_module)

    shim.install(mode="reactive", rerank_interval=1)
    cfg = FakeConfig()
    cfg.unstable_heads = [[1, 2], [3, 4]]
    cfg.rerank_frequency = 16
    cfg._build()

    assert len(build_calls) == 1
    # After the shim's pre-hook, original _build sees the overridden values.
    _, seen_unstable, seen_freq = build_calls[0]
    assert seen_unstable == []
    assert seen_freq == 1
    assert getattr(FakeConfig._build, "_wakekv_shim", False)

    # Uninstall restores the original method.
    shim.uninstall()
    assert not hasattr(FakeConfig._build, "_wakekv_shim")


def test_install_is_idempotent(monkeypatch):
    """Calling install twice should not stack wrappers."""

    class FakeConfig:
        def _build(self):
            pass

    fake_module = SimpleNamespace(FlexiCacheConfig=FakeConfig)
    import sys
    monkeypatch.setitem(sys.modules, "vllm.v1.flexicache.config", fake_module)

    shim.install(mode="reactive", rerank_interval=1)
    first = FakeConfig._build
    shim.install(mode="reactive", rerank_interval=8)
    second = FakeConfig._build

    # Same wrapper depth: __wrapped__ points to the pristine method, not a stack.
    assert second.__wrapped__ is first.__wrapped__
    # And the new interval took effect.
    assert shim._current_rerank_interval == 8
