"""The trained model: implicit-feedback matrix factorisation (ALS) over 1.5M users' liked games.

Confidence grows with hours played (1 + alpha * log1p(hours)). A new user is not trained in: their vector is
solved from their library against the fixed game vectors (fold-in), so one profile link is enough.
"""
from typing import Sequence

import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from recs.core import Library, top_up
from recs.data import Interactions


class ALS:
    name = "als"

    def __init__(self, train: Interactions, factors: int = 64, regularization: float = 0.05, alpha: float = 4.0,
                 iterations: int = 15, seed: int = 42, fallback: Sequence[int] = ()):
        self.item_ids, self.index, self.fallback = train.item_ids, train.item_index, fallback
        weights = train.matrix.copy()
        weights.data = np.log1p(weights.data)
        with threadpool_limits(1, "blas"):  # implicit runs its own threads; nested BLAS threads slow it badly
            self.model = AlternatingLeastSquares(factors=factors, regularization=regularization, alpha=alpha,
                                                 iterations=iterations, random_state=seed)
            self.model.fit(weights, show_progress=False)

    def _row(self, library: Library) -> csr_matrix | None:
        known = [(self.index[a], np.log1p(h)) for a, h in library.items() if a in self.index]
        if not known:
            return None
        cols, vals = zip(*known)
        return csr_matrix((np.array(vals, dtype=np.float32), ([0] * len(cols), cols)), shape=(1, len(self.item_ids)))

    def ranking(self, library: Library, n: int) -> list[int]:
        """The n best games not owned, best first; games the model never saw are ignored."""
        row = self._row(library)
        if row is None:
            return []
        ids, _ = self.model.recommend(0, row, N=n, filter_already_liked_items=True, recalculate_user=True)
        return [int(self.item_ids[i]) for i in ids]

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)
