import math

from eval import metrics
from recs.core import top_up


def test_recall_and_ndcg():
    assert metrics.recall_at_k([5, 1, 9], [1, 2], k=3) == 0.5
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(metrics.ndcg_at_k([5, 1, 9], [1, 2], k=3), expected)


def test_recall_is_capped_by_k():
    assert metrics.recall_at_k([1, 2], [1, 2, 3, 4], k=2) == 1.0


def test_coverage_and_popularity_rank():
    assert metrics.coverage([[1, 2], [2, 3]]) == 3
    assert metrics.mean_pop_rank([[1, 2], [2, 3]], {1: 1, 2: 5, 3: 9}) == 5.0


def test_top_up_drops_owned_and_fills():
    assert top_up([3, 3, 1], [1, 2, 4, 5], {1: 2.0}, 3) == [3, 2, 4]


def test_cold_start_scores_only_new_hidden_items():
    entries = [{"user": 1, "visible": {}, "hidden": [1, 3]}, {"user": 2, "visible": {}, "hidden": [1, 2]}]
    assert metrics.cold_start_entries(entries, new_apps={3}) == [{"user": 1, "visible": {}, "hidden": [3]}]


def test_paired_bootstrap_brackets_the_mean_gap():
    same = metrics.paired_bootstrap([0.1, 0.5, 0.3], [0.1, 0.5, 0.3])
    assert same == (0.0, 0.0, 0.0)
    gap, lo, hi = metrics.paired_bootstrap([0.0] * 50 + [0.5] * 50, [0.5] * 100)
    assert math.isclose(gap, 0.25) and 0 < lo < 0.25 < hi
