from recs.data import Catalog
from recs.explain import Explainer, grounded

CAT = Catalog({1: "Deep Rock Galactic", 2: "HELLDIVERS™ 2", 3: "Portal 2", 4: "Vermintide 2"},
              {a: ["Co-op"] for a in range(1, 5)}, {a: "2020-01-01" for a in range(1, 5)})


class FakeRanker:
    def recommend(self, library, k=10):
        return [a for a in [3, 4] if a not in library][:k]


def test_order_comes_from_the_ranker_and_reasons_from_the_llm():
    fake = lambda prompt, sch: {"reasons": [{"app_id": 4, "reason": "Like Deep Rock Galactic"},
                                            {"app_id": 3, "reason": "Puzzle break"}]}
    out = Explainer(FakeRanker(), CAT, generate=fake).explain({1: 100.0}, k=2)
    assert out == [(3, "Puzzle break"), (4, "Like Deep Rock Galactic")]


def test_schema_only_allows_the_picks_and_a_failure_keeps_the_order():
    seen = {}
    def boom(prompt, sch):
        seen.update(sch)
        raise TimeoutError
    e = Explainer(FakeRanker(), CAT, generate=boom)
    assert e.explain({1: 100.0}, k=2) == [(3, ""), (4, "")] and e.failures == 1
    assert seen["properties"]["reasons"]["items"]["properties"]["app_id"]["enum"] == [3, 4]


def test_grounded_means_the_reason_names_a_played_game():
    assert grounded("Since you loved Helldivers 2's co-op", {2: 50.0}, CAT)
    assert not grounded("A great puzzle game", {2: 50.0}, CAT)
