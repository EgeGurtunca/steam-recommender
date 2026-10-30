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


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else 0.0


def diversify(ranked: list[int], tags: dict, lam: float, k: int = 10, top: int = 5) -> list[int]:
    """Maximal marginal relevance: pick k from a ranking, each step weighing its place in the ranking (lam) against
    its tag overlap with what is already picked (1 - lam). lam 1 keeps the ranking."""
    n, picked, left = len(ranked), [], list(ranked)
    place = {a: 1 - i / n for i, a in enumerate(ranked)}
    while left and len(picked) < k:
        best = max(left, key=lambda a: lam * place[a] - (1 - lam) * max(
            (jaccard(tags.get(a, [])[:top], tags.get(p, [])[:top]) for p in picked), default=0.0))
        picked.append(best)
        left.remove(best)
    return picked
