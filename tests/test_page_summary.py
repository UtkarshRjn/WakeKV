import numpy as np

from wakekv.page_summary import prefetch_recall


def test_causal_page_max_recalls_the_previous_hot_page():
    # Two steps, one layer, one head, four pages. Top-1 index is token 0
    # (page 0) then token 16 (page 1). Previous page-max points at page 1.
    page_max = np.zeros((2, 1, 1, 4), dtype=np.float16)
    page_max[0, 0, 0, 1] = 1.0
    page_max[1, 0, 0, 0] = 1.0
    topk = np.full((2, 1, 1, 1), -1, dtype=np.int32)
    topk[0, 0, 0, 0] = 0
    topk[1, 0, 0, 0] = 16
    out = prefetch_recall(page_max, topk, page_size=16, budget=1)
    assert out["causal_page_max_recall"] == 1.0
    assert out["same_step_page_max_recall"] == 0.0
