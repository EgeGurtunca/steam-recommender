"""Two stages, as large recommenders are built: EASE proposes 50 games, a small gradient-boosted model reorders them
with what EASE does not see (how popular, how new, how well reviewed, how close in tags).

The model learns from users EASE never trained on: their EASE scores look like a new user's, not inflated by
having seen the games that were hidden from them.
"""
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from recs.core import Library, top_up

CANDIDATES = 50
FEATURES = ["ease score", "ease score / best", "ease rank", "log popularity", "year", "positive share", "log reviews",
            "tag similarity", "log library size"]


class Features:
    """One row per candidate. store and the tag model may be missing a game: those features read as NaN, which
    gradient boosting handles as its own branch."""

    def __init__(self, ease, content, popularity: dict[int, int], store: dict, released: dict[int, str]):
        self.ease, self.content, self.popularity, self.store, self.released = ease, content, popularity, store, released

    def candidates(self, library: Library, n: int = CANDIDATES) -> tuple[list[int], np.ndarray]:
        scores = self.ease.scores(library)
        if scores is None:
            return [], np.empty((0, len(FEATURES)))
        n = min(n, len(scores))
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top], kind="stable")]
        top = top[np.isfinite(scores[top])]  # owned games are -inf
        apps = [int(self.ease.item_ids[i]) for i in top]
        tags = self.content.scores(library) if self.content else None
        rows, best = [], scores[top[0]] if len(top) and scores[top[0]] > 0 else 1.0
        for rank, (app, i) in enumerate(zip(apps, top)):
            v = self.store.get(app) or {}
            pos, neg = v.get("positive", 0), v.get("negative", 0)
            year = (v.get("released") or self.released.get(app) or "")[:4]
            t = self.content.index.get(app) if self.content else None
            rows.append([scores[i], scores[i] / best, rank, math.log1p(self.popularity.get(app, 0)),
                         float(year) if year else np.nan, pos / (pos + neg) if pos + neg else np.nan,
                         math.log1p(pos + neg),
                         tags[t] if tags is not None and t is not None else np.nan, math.log1p(len(library))])
        return apps, np.asarray(rows, dtype=np.float64)


def training_rows(features: Features, entries: list[dict], progress=None) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for n, e in enumerate(entries, 1):
        apps, x = features.candidates(e["visible"])
        hidden = set(e["hidden"])
        xs.append(x)
        ys.append(np.array([a in hidden for a in apps], dtype=np.int8))
        if progress and n % 2000 == 0:
            progress(n, len(entries))
    x = np.vstack(xs)
    x[:, np.isnan(x).all(axis=0)] = 0.0  # a column with no value at all breaks sklearn's binning
    return x, np.concatenate(ys)


class Ranker:
    name = "ease+ranker"

    def __init__(self, features: Features, fallback=(), seed: int = 0):
        self.features, self.fallback = features, fallback
        self.model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping="auto",
                                                    random_state=seed)

    def fit(self, entries: list[dict], progress=None) -> "Ranker":
        x, y = training_rows(self.features, entries, progress)
        self.model.fit(x, y)
        return self

    def ranking(self, library: Library, n: int) -> list[int]:
        apps, x = self.features.candidates(library)
        if not apps:
            return []
        p = self.model.predict_proba(x)[:, 1]
        return [apps[i] for i in np.argsort(-p, kind="stable")][:n]

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)
