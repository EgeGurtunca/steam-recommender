"""One player, from their own Steam profile link: what they like (tags), what to play next (EASE), and which
2024-2026 games fit (tags times popularity, from the 2026 store catalogue the 2022 model never saw).

Run: python -m recs.profile https://steamcommunity.com/id/<name>
"""
import math
import sys
from collections import Counter

from recs.core import Library

# Steam genres that mean a program, not a game (Wallpaper Engine, editors, courses)
NON_GAME = {"Utilities", "Design & Illustration", "Video Production", "Animation & Modeling", "Audio Production",
            "Web Publishing", "Photo Editing", "Software Training", "Accounting", "Education", "Game Development"}
TAGS_USED = 10
# how a game is played, not what kind of game it is: left out of the taste, reported by social_split instead
FORMAT = {"Multiplayer", "Singleplayer", "Free to Play", "Co-op", "Online Co-Op", "Local Co-Op", "Massively Multiplayer",
          "PvP", "Online PvP", "PvE", "Early Access", "Indie", "Casual"}
SOCIAL = {"Multiplayer", "Co-op", "Online Co-Op", "Local Co-Op", "Massively Multiplayer", "PvP", "Online PvP"}  # Steam lists a game's tags by votes; the first ones describe it best


def is_game(app: int, store: dict) -> bool:
    return not (set((store.get(app) or {}).get("genres", [])) & NON_GAME)


def idf(store: dict) -> dict[str, float]:
    df = Counter(t for v in store.values() for t in set(v["tags"][:TAGS_USED]))
    return {t: math.log(len(store) / n) for t, n in df.items()}


def tag_vector(library: Library, store: dict, weights: dict[str, float]) -> Counter:
    """sqrt(hours) times how rare each tag is. log1p squashed hours so hard that 100 short games beat three
    2000-hour ones; sqrt keeps a 1000-hour game ten times a 10-hour one without letting one idle game drown all."""
    vec = Counter()
    for app, hours in library.items():
        for t in (store.get(app) or {}).get("tags", [])[:TAGS_USED]:
            vec[t] += math.sqrt(hours) * weights.get(t, 0.0)
    return vec


def engagement(library: Library, store: dict, mode: str = "rel-log") -> dict[int, float]:
    """How much more than a typical owner this player played each game: 1000 hours of a game whose median is
    40 says more than 1000 hours of an idle game whose median is 80. Games without a median count 1."""
    out = {}
    for app, hours in library.items():
        median = (store.get(app) or {}).get("median_hours", 0)
        if mode == "sqrt-hours":
            out[app] = math.sqrt(hours)
        elif median > 0:
            out[app] = math.log1p(hours / median) if mode == "rel-log" else math.sqrt(hours / median)
    return out


def taste_profile(library: Library, store: dict, top: int = 12, weights: dict | None = None) -> list[tuple[str, float]]:
    vec = tag_vector(library, store, weights or idf(store))
    for t in FORMAT:
        vec.pop(t, None)
    total = sum(vec.values()) or 1.0
    return [(t, v / total) for t, v in vec.most_common(top)]


def tag_edges(tags: list[str], store: dict, min_positive: int = 500, min_jaccard: float = 0.08) -> list[tuple[str, str, float]]:
    """Which of these tags tend to describe the same games (Jaccard over reviewed games), for the bubble graph."""
    games = {t: set() for t in tags}
    for app, v in store.items():
        if v.get("positive", 0) >= min_positive:
            for t in v["tags"][:TAGS_USED]:
                if t in games:
                    games[t].add(app)
    edges = []
    for i, a in enumerate(tags):
        for b in tags[i + 1:]:
            union = len(games[a] | games[b])
            j = len(games[a] & games[b]) / union if union else 0.0
            if j >= min_jaccard:
                edges.append((a, b, round(j, 3)))
    return edges


def social_split(library: Library, store: dict) -> dict[str, float]:
    """Share of hours in games tagged multiplayer or co-op, against the rest (a proxy for "with friends")."""
    social = sum(h for a, h in library.items() if set((store.get(a) or {}).get("tags", [])[:TAGS_USED]) & SOCIAL)
    total = sum(library.values()) or 1.0
    return {"multiplayer": social / total, "singleplayer": 1 - social / total}


def new_releases(library: Library, store: dict, since: str = "2024-01-01", k: int = 10, min_positive: int = 2000,
                 weights: dict | None = None, pop_power: float = 1.0) -> list[int]:
    """Games released from `since`, not owned, with at least `min_positive` positive reviews, ranked by tag
    similarity to the player times log1p(positive reviews) ** pop_power."""
    weights = weights or idf(store)
    user = tag_vector(library, store, weights)
    norm_u = math.sqrt(sum(v * v for v in user.values())) or 1.0
    scored = []
    for app, v in store.items():
        if app in library or not v["released"] or v["released"] < since or v["positive"] < min_positive or not is_game(app, store):
            continue
        tags = v["tags"][:TAGS_USED]
        norm_c = math.sqrt(sum(weights.get(t, 0) ** 2 for t in tags)) or 1.0
        cos = sum(user.get(t, 0) * weights.get(t, 0) for t in tags) / (norm_u * norm_c)
        scored.append((cos * math.log1p(v["positive"]) ** pop_power, app))
    return [app for _, app in sorted(scored, reverse=True)[:k]]


def main(link: str) -> None:
    from recs import steam
    from recs.data import BUILD_DIR, load_catalog, load_store
    from recs.ease import EASE

    lib, store, catalog = steam.load_library(link), load_store(), load_catalog()
    played = {a: h for a, h in lib.games.items() if h >= 2}
    weights = idf(store)
    print(f"{len(lib.games)} games, {len(played)} played for 2 hours or more\n\nYour taste:")
    for t, share in taste_profile(played, store, weights=weights):
        print(f"  {t:<24} {'#' * round(share * 100)} {share:.0%}")
    ease = EASE.load_or_build(BUILD_DIR / "ease")
    from recs.web import next_picks, page_ranker
    _, picks = next_picks(lib, store, page_ranker(ease))  # as the page ranks
    split = social_split(played, store)
    print(f"\nHours in multiplayer or co-op games: {split['multiplayer']:.0%}, the rest: {split['singleplayer']:.0%}")
    print("\nPlay next (EASE + the order you played in, from 1.5M players up to 2022):")
    for i, a in enumerate(picks, 1):
        print(f"  {i:2}. {catalog.titles.get(a) or store.get(a, {}).get('title', a)}")
    print("\nNew since 2024 that fit your tags (2026 store catalogue):")
    for i, a in enumerate(new_releases(played, store, weights=weights), 1):
        print(f"  {i:2}. {store[a]['title']} ({store[a]['released'][:4]})")


if __name__ == "__main__":
    main(sys.argv[1])
