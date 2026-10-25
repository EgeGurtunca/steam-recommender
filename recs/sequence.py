"""Order in a library: what players liked next. EASE sees a library as a set; these count, over likes in date
order, which game tends to follow which, and nudge EASE's candidates with it. On the time split
(eval/time_split.py) EASE + transitions found 14% more than EASE on test.

A library's values here are recency weights (the latest games near 1), not hours: see eval.time_split.by_recency
and recs.web, which reads them from Steam's last-played times.
"""
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, load_npz, save_npz

from recs.core import top_up

GAMMA = 3.0  # best of 0.3 to 30 on the time split's val users
HALF_LIFE = 180.0  # days: how fast an older game's say fades


def transitions(users: np.ndarray, apps: np.ndarray, dates: np.ndarray, index: dict[int, int], window: int = 3) -> csr_matrix:
    """How often each game was followed, among a user's likes in date order, by each of the next `window` likes.
    Rows sum to 1: P(liked soon after | this game). Games outside index are skipped."""
    order = np.lexsort((dates, users))
    u, a = users[order], apps[order]
    keys = np.array(sorted(index), dtype=a.dtype)
    vals = np.array([index[int(k)] for k in keys], dtype=np.int64)
    pos = np.clip(np.searchsorted(keys, a), 0, len(keys) - 1)  # vectorised lookup: 15M likes, a dict per like is slow
    known = keys[pos] == a
    u, cols = u[known], vals[pos[known]]
    rows_, cols_ = [], []
    for k in range(1, window + 1):
        same = u[:-k] == u[k:]
        rows_.append(cols[:-k][same])
        cols_.append(cols[k:][same])
    r, c = np.concatenate(rows_), np.concatenate(cols_)
    t = csr_matrix((np.ones(len(r), dtype=np.float32), (r, c)), shape=(len(index), len(index)))
    sums = np.asarray(t.sum(axis=1)).ravel()
    sums[sums == 0] = 1.0
    return csr_matrix(t.multiply(1 / sums[:, None]))


class Markov:
    """What players liked next: the user's games, weighted toward the latest (by_recency), times transitions()."""

    def __init__(self, t: csr_matrix, item_ids: np.ndarray, index: dict[int, int], fallback=(), name: str = "markov"):
        self.t, self.item_ids, self.index, self.fallback, self.name = t, item_ids, index, fallback, name

    def scores(self, library) -> np.ndarray | None:
        apps = [a for a in library if a in self.index]
        if not apps:
            return None
        cols = [self.index[a] for a in apps]
        s = np.asarray(self.t[cols].T @ np.array([library[a] for a in apps], dtype=np.float32)).ravel()
        s[cols] = -np.inf
        return s

    def recommend(self, library, k: int = 10) -> list[int]:
        s = self.scores(library)
        top = [] if s is None else [int(self.item_ids[i]) for i in np.argsort(-s)[:k] if s[i] > 0]
        return top_up(top, self.fallback, library, k)


class MarkovNudge:
    """EASE's top 100, each moved up by its transition score from the user's latest games."""

    def __init__(self, ease, markov: Markov, gamma: float, fallback=(), n: int = 100):
        self.ease, self.markov, self.gamma, self.fallback, self.n = ease, markov, gamma, fallback, n
        self.name = f"ease+markov:{gamma:g}"

    def ranking(self, library, n: int) -> list[int]:
        cands = self.ease.ranking(library, max(n, self.n))
        m = self.markov.scores(library)
        nxt = {a: max(float(m[self.markov.index[a]]), 0.0) if m is not None and a in self.markov.index else 0.0
               for a in cands}
        top = max(nxt.values(), default=0.0) or 1.0
        score = {a: 1 - i / len(cands) + self.gamma * nxt[a] / top for i, a in enumerate(cands)}
        return sorted(cands, key=lambda a: -score[a])[:n]

    def recommend(self, library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)


def load_transitions(index: dict[int, int], cache: Path) -> csr_matrix:
    """transitions() over every positive (minutes from the raw reviews the first time, then a file)."""
    if cache.exists():
        t = load_npz(cache)
        if t.shape == (len(index), len(index)):  # built for another game list (EASE rebuilt): count again
            return t
    from recs.data import load_positive_dates
    users, apps, _, dates = load_positive_dates()
    t = transitions(users, apps, dates, index)
    save_npz(cache, t)
    return t
