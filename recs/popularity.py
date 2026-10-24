"""The floor every other system has to clear: the games the most training users liked."""
import numpy as np

from recs.core import Library, top_up
from recs.data import Interactions


class Popularity:
    name = "popularity"

    def __init__(self, train: Interactions):
        users_per_item = np.asarray((train.matrix > 0).sum(axis=0)).ravel()
        order = np.lexsort((train.item_ids, -users_per_item))  # most liked first, ties by app id
        self.ranking: list[int] = train.item_ids[order].tolist()

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return top_up([], self.ranking, library, k)
