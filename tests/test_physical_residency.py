"""Tests for GQA-aware physical residency."""

from __future__ import annotations

from wakekv.physical_residency import (
    aggregate_stream_to_physical,
    kv_bytes,
    query_to_kv_map,
    simulate_physical,
)


def test_query_to_kv_map_mistral_style():
    # 32 Q heads, 8 KV heads → groups of 4
    m = query_to_kv_map(32, 8)
    assert len(m) == 32
    assert m[0] == 0 and m[3] == 0 and m[4] == 1
    assert m[31] == 7


def test_aggregate_unions_query_groups():
    # 1 layer, 4 Q, 2 KV. Step0: q0 wants {1}, q1 wants {2} → kv0={1,2}
    stream = [[frozenset({1}), frozenset({2}), frozenset({3}), frozenset()]]
    phys, n = aggregate_stream_to_physical(stream, n_layers=1, n_q_heads=4, n_kv_heads=2)
    assert n == 2
    assert phys[0][0] == frozenset({1, 2})
    assert phys[0][1] == frozenset({3})


def test_kv_bytes_formula():
    # 2 * P * d * b * pages
    assert kv_bytes(10, page_tokens=16, head_dim=128, dtype_bytes=2) == 2 * 16 * 128 * 2 * 10


def test_simulate_physical_reactive_misses_less_or_equal_evict():
    # Synthetic stream: revisit old pages after eviction pressure.
    stream = []
    for t in range(20):
        # alternate between page sets that force churn
        wanted = frozenset({t % 8, (t + 1) % 8, (t + 2) % 8})
        stream.append([wanted, wanted, wanted, wanted])  # 1 layer, 4 Q, will map to 2 KV
    react = simulate_physical(
        stream, n_layers=1, n_q_heads=4, n_kv_heads=2, policy="reactive", budget=2,
        page_tokens=16, head_dim=64,
    )
    evict = simulate_physical(
        stream, n_layers=1, n_q_heads=4, n_kv_heads=2, policy="evict", budget=2,
        page_tokens=16, head_dim=64,
    )
    assert react.logical.miss_rate <= evict.logical.miss_rate + 1e-9
    assert react.peak_kv_bytes() > 0
