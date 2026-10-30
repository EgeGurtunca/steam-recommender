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

    def __init__(self, train: Interactions, lam: float = 500.0, fallback: Sequence[int] = (), binary: bool = True):
        """binary=False learns from the matrix values as weights (eval/time_split.py: recent likes weigh more)."""
        self.item_ids, self.index, self.fallback = train.item_ids, train.item_index, fallback
        x = (train.matrix > 0).astype(np.float32) if binary else train.matrix.astype(np.float32)
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

    def save_small(self, path, k: int = 200) -> None:
        """Each game's k largest weights by size, in float16: 8 MB instead of 447. On the time split it found the
        same as the full matrix (eval/time_split.py, ease-top:200); dropping the negative weights did not."""
        idx = np.argpartition(-np.abs(self.b), k, axis=1)[:, :k].astype(np.int16)
        np.savez_compressed(path, idx=idx, w=np.take_along_axis(self.b, idx, axis=1).astype(np.float16),
                            items=self.item_ids)

    @classmethod
    def load_small(cls, path) -> "EASE":
        f = np.load(path)
        e = cls.__new__(cls)
        e.item_ids, e.fallback = f["items"], ()
        e.b = np.zeros((len(e.item_ids),) * 2, dtype=np.float32)
        np.put_along_axis(e.b, f["idx"].astype(np.int64), f["w"].astype(np.float32), axis=1)
        e.index = {int(a): i for i, a in enumerate(e.item_ids)}
        return e

    @classmethod
    def load_or_build(cls, folder) -> "EASE":
        """The page's weights: the full matrix if built, else the small copy (recs/bundle.py), else trained on every
        positive (about 20 s) and saved."""
        from pathlib import Path
        from recs import data
        from recs.popularity import Popularity
        if (Path(folder) / "b.npy").exists():
            return cls.load(folder)
        if (Path(folder) / "small.npz").exists():
            return cls.load_small(Path(folder) / "small.npz")
        train = data.build_interactions(*data.load_positives(), exclude_users=set())
        ease = cls(train, fallback=Popularity(train).ranking)
        ease.save(folder)
        return ease

    @classmethod
    def load(cls, folder, fallback: Sequence[int] = ()) -> "EASE":
        from pathlib import Path
        e = cls.__new__(cls)
        e.b, e.item_ids, e.fallback = np.load(Path(folder) / "b.npy"), np.load(Path(folder) / "items.npy"), fallback
        e.index = {int(a): i for i, a in enumerate(e.item_ids)}
        return e

    def scores(self, library: Library) -> np.ndarray | None:
        """A score for every game (columns in item_ids order), owned ones -inf; None if no game is known."""
        apps = [a for a in library if a in self.index]
        if not apps:
            return None
        cols = [self.index[a] for a in apps]
        weigh = getattr(self, "weigh", None)  # None: every liked game counts once, as in training
        w = weigh(library) if weigh else {}
        scores = np.asarray([w.get(a, 1.0) for a in apps], dtype=np.float32) @ self.b[cols]
        scores[cols] = -np.inf
        return scores

    def because(self, library: Library, picks: list[int]) -> dict[int, int]:
        """For each pick, the owned game with the biggest term in its score: "because you played X", exact and
        free, since a score is a sum over the library. Picks EASE does not know get no entry."""
        owned = [a for a in library if a in self.index]
        if not owned:
            return {}
        rows = self.b[[self.index[a] for a in owned]]
        return {p: owned[int(np.argmax(rows[:, self.index[p]]))] for p in picks if p in self.index}

    def ranking(self, library: Library, n: int) -> list[int]:
        scores = self.scores(library)
        if scores is None:
            return []
        top = np.argpartition(-scores, n)[:n]
        return [int(self.item_ids[i]) for i in top[np.argsort(-scores[top], kind="stable")]]

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)
