import numpy as np

from recs.content import Content
from recs.data import Catalog, build_interactions
from recs.popularity import Popularity


def catalog(tags: dict[int, list[str]]) -> Catalog:
    return Catalog({a: f"Game {a}" for a in tags}, tags, {a: "2020-01-01" for a in tags})


def test_content_prefers_shared_tags():
    c = Content(catalog({1: ["Co-op", "Shooter"], 2: ["Co-op", "Shooter", "PvE"], 3: ["Puzzle"]}))
    assert c.recommend({1: 50.0}, 1) == [2]


def test_content_ignores_unknown_and_untagged():
    c = Content(catalog({1: ["Co-op"], 2: ["Co-op", "PvE"], 3: ["Puzzle"], 4: []}), fallback=[4, 3, 2])
    recs = c.recommend({1: 50.0, 999: 10.0}, 3)
    assert len(recs) == 3 and recs[0] == 2 and 1 not in recs and 999 not in recs


def test_content_empty_library_uses_fallback():
    c = Content(catalog({1: ["Co-op"], 2: ["PvE"]}), fallback=[2, 1])
    assert c.recommend({}, 2) == [2, 1]


def test_content_weighs_long_played_games_more():
    c = Content(catalog({1: ["Racing"], 2: ["Horror"], 3: ["Racing", "Sim"], 4: ["Horror", "Sim"]}))
    assert c.recommend({1: 300.0, 2: 3.0}, 1) == [3]


def interactions(pairs):
    """pairs: (user, app) with 5 hours each."""
    users, apps = zip(*pairs)
    return build_interactions(np.array(users), np.array(apps), np.full(len(pairs), 5.0), exclude_users=set())


def two_groups():
    """Users 1-20 played games 1 and 2, users 21-40 played games 3 and 4."""
    pairs = [(u, a) for u in range(1, 21) for a in (1, 2)] + [(u, a) for u in range(21, 41) for a in (3, 4)]
    return interactions(pairs)


def test_als_recommends_co_played_game():
    from recs.als import ALS
    als = ALS(two_groups(), factors=4, iterations=10)
    assert als.recommend({1: 10.0}, 1) == [2]
    assert als.recommend({3: 10.0}, 1) == [4]


def test_als_unknown_only_library_falls_back():
    from recs.als import ALS
    als = ALS(two_groups(), factors=4, iterations=10, fallback=[3, 1, 2])
    assert als.recommend({999: 5.0}, 2) == [3, 1]
    assert als.recommend({}, 2) == [3, 1]


def test_popularity_orders_by_users_and_skips_owned():
    train = interactions([(1, 7), (2, 7), (3, 7), (1, 8), (2, 8), (1, 9)])
    pop = Popularity(train)
    assert pop.ranking == [7, 8, 9]
    assert pop.recommend({7: 5.0}, 2) == [8, 9]


class Stub:
    def __init__(self, ranked):
        self.ranked = ranked

    def ranking(self, library, n):
        return [a for a in self.ranked if a not in library][:n]


def test_hybrid_interleaves_without_duplicates():
    from recs.hybrid import Hybrid
    h = Hybrid(Stub([1, 2, 3]), Stub([2, 4, 5]), n_candidates=4)
    assert h.candidates({}) == [1, 2, 4, 3]


def test_hybrid_never_returns_owned_or_short():
    from recs.hybrid import Hybrid
    h = Hybrid(Stub([9, 1]), Stub([9, 2]), n_candidates=5, fallback=[9, 3, 4, 5, 6])
    out = h.candidates({9: 5.0})
    assert 9 not in out and len(out) == 5 and len(set(out)) == 5
    assert h.recommend({9: 5.0}, 2) == out[:2]


def test_hybrid_without_content_uses_als_only():
    from recs.hybrid import Hybrid
    assert Hybrid(Stub([1, 2, 3]), None, n_candidates=2).candidates({}) == [1, 2]


def test_ease_recommends_co_played_game():
    from recs.ease import EASE
    e = EASE(two_groups(), lam=1.0)
    assert e.recommend({1: 10.0}, 1) == [2]
    assert e.recommend({3: 10.0}, 1) == [4]


def test_ease_unknown_only_library_falls_back():
    from recs.ease import EASE
    assert EASE(two_groups(), lam=1.0, fallback=[3, 1]).recommend({999: 5.0}, 2) == [3, 1]


def test_content_prior_breaks_ties_toward_popular_games():
    c = Content(catalog({1: ["Co-op"], 2: ["Co-op", "PvE"], 3: ["Co-op", "PvE"]}), prior={2: 10.0, 3: 1000.0}, beta=1.0)
    assert c.recommend({1: 50.0}, 1) == [3]
    assert Content(catalog({1: ["Co-op"], 2: ["Co-op", "PvE"], 3: ["Co-op", "PvE"]})).recommend({1: 50.0}, 1) == [2]


def test_ease_survives_a_save_and_load(tmp_path):
    from recs.ease import EASE
    e = EASE(two_groups(), lam=1.0, fallback=[3, 1])
    e.save(tmp_path / "ease")
    back = EASE.load(tmp_path / "ease", fallback=[3, 1])
    assert back.recommend({1: 10.0}, 1) == [2] and back.recommend({999: 1.0}, 2) == [3, 1]


def test_warm_start_adds_back_only_each_new_games_first_k_reviews():
    from eval.warm_start import with_first
    base = (np.array([1, 2]), np.array([10, 10]), np.array([5.0, 5.0]))
    early = (np.array([1, 2, 9]), np.array([20, 20, 20]), np.array([3.0, 3.0, 3.0]), np.array([1, 3, 2]))
    train = with_first(2, base, early, eval_users={9})
    assert train.matrix[:, train.item_index[20]].nnz == 1  # order 1 kept, order 3 too late, eval user 9 never


def test_ranker_reorders_ease_candidates_by_what_it_learned():
    from recs.ranker import FEATURES, Features, Ranker

    class FakeEase:  # five games, EASE prefers 1 > 2 > 3; game 3 is the one users actually pick
        item_ids = np.array([1, 2, 3, 4, 5])

        def scores(self, library):
            s = np.array([5.0, 4.0, 3.0, 0.0, 0.0])
            s[[int(a) - 1 for a in library if a in (1, 2, 3, 4, 5)]] = -np.inf
            return s

    store = {3: {"positive": 9000, "negative": 100, "released": "2021-01-01"}}
    f = Features(FakeEase(), None, {1: 10, 2: 10, 3: 10}, store, {})
    apps, x = f.candidates({4: 5.0}, n=3)
    assert apps == [1, 2, 3] and x.shape == (3, len(FEATURES))
    r = Ranker(f).fit([{"visible": {4: 5.0}, "hidden": [3]}, {"visible": {5: 5.0}, "hidden": [3]}] * 20)
    assert r.recommend({4: 5.0}, 1) == [3]
