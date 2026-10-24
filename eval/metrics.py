"""Ranking metrics for one user's top-k against the games hidden from that user."""
import math


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
