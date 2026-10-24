"""EASE (Steck, 2019): a game-to-game weight matrix with a closed-form solution, B = I - P / diag(P) with
P = (X'X + lam I)^-1, diagonal forced to zero. One matrix inverse (10.8K x 10.8K), no iterations, and on data
like this it often beats ALS. A user's scores are their liked-games row times B."""
from typing import Sequence

import numpy as np
from threadpoolctl import threadpool_limits

from recs.core import Library, top_up
from recs.data import Interactions


class EASE:
    name = "ease"

    def __init__(self, train: Interactions, lam: float = 500.0, fallback: Sequence[int] = ()):
        self.item_ids, self.index, self.fallback = train.item_ids, train.item_index, fallback
        x = (train.matrix > 0).astype(np.float32)
        gram = (x.T @ x).toarray().astype(np.float64)
        gram[np.diag_indices_from(gram)] += lam
        with threadpool_limits(None, "blas"):  # the package pins BLAS to 1 thread for implicit; the inverse wants all
            p = np.linalg.inv(gram)
        self.b = (p / -np.diag(p)).astype(np.float32)
        np.fill_diagonal(self.b, 0.0)

    def save(self, folder) -> None:
        """The 10.8K x 10.8K weights take minutes to build and half a second to load."""
        from pathlib import Path
        Path(folder).mkdir(parents=True, exist_ok=True)
        np.save(Path(folder) / "b.npy", self.b)
        np.save(Path(folder) / "items.npy", self.item_ids)

    @classmethod
    def load(cls, folder, fallback: Sequence[int] = ()) -> "EASE":
        from pathlib import Path
        e = cls.__new__(cls)
        e.b, e.item_ids, e.fallback = np.load(Path(folder) / "b.npy"), np.load(Path(folder) / "items.npy"), fallback
        e.index = {int(a): i for i, a in enumerate(e.item_ids)}
        return e

    def ranking(self, library: Library, n: int) -> list[int]:
        cols = [self.index[a] for a in library if a in self.index]
        if not cols:
            return []
        scores = self.b[cols].sum(axis=0)
        scores[cols] = -np.inf
        top = np.argpartition(-scores, n)[:n]
        return [int(self.item_ids[i]) for i in top[np.argsort(-scores[top], kind="stable")]]

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)
