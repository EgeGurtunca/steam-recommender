"""Ranking metrics for one user's top-k against the games hidden from that user."""
import math

import numpy as np

from recs.core import diversify, jaccard  # noqa: F401  (diversify lives with the page's code)


def recall_at_k(recs: list[int], hidden: list[int], k: int = 10) -> float:
    """Hidden games found in the top k, out of the most that could fit there."""
    return len(set(recs[:k]) & set(hidden)) / min(k, len(hidden))


def ndcg_at_k(recs: list[int], hidden: list[int], k: int = 10) -> float:
    hits = set(hidden)
    dcg = sum(1 / math.log2(i + 2) for i, app in enumerate(recs[:k]) if app in hits)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(hidden))))
    return dcg / ideal


def coverage(all_recs: list[list[int]]) -> int:
    """How many different games the system recommended across all users."""
    return len({app for recs in all_recs for app in recs})


def mean_pop_rank(all_recs: list[list[int]], rank: dict[int, int]) -> float:
    """Average popularity rank of the recommended games (1 = most popular): low means popularity bias."""
    ranks = [rank.get(app, len(rank) + 1) for recs in all_recs for app in recs]
    return sum(ranks) / len(ranks)


def cold_start_entries(entries: list[dict], new_apps: set[int]) -> list[dict]:
    """Cold-start scoring: only the hidden games that are "new", only for users who have one."""
    out = []
    for e in entries:
        hidden = [a for a in e["hidden"] if a in new_apps]
        if hidden:
            out.append({**e, "hidden": hidden})
    return out


def paired_bootstrap(base: list[float], other: list[float], n: int = 10000,
                     seed: int = 0) -> tuple[float, float, float]:
    """Mean of other - base over the same users, with a 95% interval from resampling users. If the interval
    holds 0, the eval cannot tell the two systems apart."""
    d = np.asarray(other, dtype=float) - np.asarray(base, dtype=float)
    means = d[np.random.default_rng(seed).integers(0, len(d), (n, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def tag_diversity(all_recs: list[list[int]], tags: dict, top: int = 5) -> float:
    """1 minus the mean tag overlap (Jaccard over each game's first `top` tags) between two picks of one list."""
    pairs = [jaccard(tags.get(a, [])[:top], tags.get(b, [])[:top])
             for recs in all_recs for i, a in enumerate(recs) for b in recs[i + 1:]]
    return 1 - sum(pairs) / len(pairs) if pairs else 0.0

