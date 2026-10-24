"""The one interface every system implements, so the eval (and later a web page or a bot) can call any of them."""
from typing import Protocol, Sequence

Library = dict[int, float]  # app_id -> hours played


class Recommender(Protocol):
    name: str

    def recommend(self, library: Library, k: int = 10) -> list[int]: ...


def top_up(recs: list[int], fallback: Sequence[int], library: Library, k: int) -> list[int]:
    """recs without repeats or owned games, filled to k from fallback (usually the popularity ranking)."""
    out, seen = [], set(library)
    for app in list(recs) + list(fallback):
        if app not in seen:
            out.append(app)
            seen.add(app)
            if len(out) == k:
                break
    return out
