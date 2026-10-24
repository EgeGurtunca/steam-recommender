"""Candidates from both sides: ALS knows what similar players liked, content knows games no one has played yet.
Taken in turns, one from each, repeats and owned games skipped. The LLM reranker reorders this list."""
from itertools import chain, zip_longest
from typing import Sequence

from recs.core import Library, top_up


class Hybrid:
    name = "hybrid"

    def __init__(self, als, content, n_candidates: int = 50, fallback: Sequence[int] = ()):
        self.als, self.content, self.n, self.fallback = als, content, n_candidates, fallback

    def candidates(self, library: Library) -> list[int]:
        content = self.content.ranking(library, self.n) if self.content else []  # None: ALS candidates only
        pairs = zip_longest(self.als.ranking(library, self.n), content)
        return top_up([a for a in chain.from_iterable(pairs) if a is not None], self.fallback, library, self.n)

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return self.candidates(library)[:k]
