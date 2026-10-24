"""Games whose tags look like the tags of what you played most. Needs no other users, so it also works for games
no one in the training data has played yet (the cold-start case ALS cannot handle)."""
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from recs.core import Library, top_up
from recs.data import Catalog


class Content:
    name = "content"

    def __init__(self, catalog: Catalog, fallback: Sequence[int] = (), prior: dict[int, float] | None = None,
                 beta: float = 0.0):
        """prior: a popularity count per game (e.g. positive reviews); scores are multiplied by log1p(prior)**beta,
        so beta=0 is pure tag similarity and higher beta leans toward games many people actually play."""
        self.item_ids = np.array([a for a, tags in catalog.tags.items() if tags])
        self.index = {int(a): i for i, a in enumerate(self.item_ids)}
        self.fallback = fallback
        counts = np.array([(prior or {}).get(int(a), 0.0) for a in self.item_ids], dtype=np.float64)
        self.weight = np.log1p(counts) ** beta if beta else np.ones(len(self.item_ids))
        # rows come out L2-normalised, so a dot product with a unit profile is cosine similarity
        self.matrix = TfidfVectorizer(analyzer=lambda tags: tags).fit_transform(catalog.tags[a] for a in self.item_ids)

    def scores(self, library: Library) -> np.ndarray | None:
        """Cosine similarity of every tagged game to the user's profile, or None if no owned game has tags."""
        owned = [(self.index[a], np.log1p(h)) for a, h in library.items() if a in self.index]
        if not owned:
            return None
        rows, weights = zip(*owned)
        profile = np.asarray(self.matrix[list(rows)].T @ np.array(weights)).ravel()
        norm = np.linalg.norm(profile)
        return np.asarray(self.matrix @ (profile / norm)).ravel() * self.weight if norm else None

    def ranking(self, library: Library, n: int) -> list[int]:
        """The n best-scoring games not owned, best first."""
        s = self.scores(library)
        if s is None:
            return []
        top = np.argpartition(-s, min(n + len(library), len(s) - 1))[: n + len(library)]
        ranked = [int(self.item_ids[i]) for i in top[np.argsort(-s[top], kind="stable")]]
        return [a for a in ranked if a not in library][:n]

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)
