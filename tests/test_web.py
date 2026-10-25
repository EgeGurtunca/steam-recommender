from recs import web
from recs.data import Catalog
from recs.steam import SteamLibrary

STORE = {1: {"title": "A", "tags": ["Strategy", "Historical"], "genres": ["Strategy"], "released": "2016-01-01", "positive": 5000},
         2: {"title": "B", "tags": ["Strategy"], "genres": ["Strategy"], "released": "2025-01-01", "positive": 5000},
         3: {"title": "Tool", "tags": ["Utilities"], "genres": ["Utilities"], "released": "2019-01-01", "positive": 5000}}


class FakeRanker:
    def ranking(self, library, n):
        return [3, 1, 2]


def test_payload_has_taste_split_and_picks_without_owned_or_programs():
    lib = SteamLibrary("765", {1: 50.0, 9: 1.0}, recent={1: 3.0}, titles={1: "A", 9: "Unplayed"})
    p = web.payload(lib, STORE, FakeRanker(), Catalog({1: "A", 2: "B", 3: "Tool"}, {}, {}))
    assert p["games"] == 2 and p["played"] == 1
    assert [t["tag"] for t in p["taste"]["all"]] == ["Historical", "Strategy"] or p["taste"]["all"][0]["tag"] in {"Historical", "Strategy"}
    assert p["taste"]["recent"] and set(p["split"]) == {"multiplayer", "singleplayer"}
    assert [g["app_id"] for g in p["next"]] == [2]
    assert all(g["header"].startswith("https://") for g in p["next"])


def test_reasons_follow_the_page_order_and_use_store_titles_for_new_games():
    lib = SteamLibrary("765", {1: 50.0, 7: 30.0}, titles={1: "A", 7: "Brand New"})
    store = {**STORE, 7: {"title": "Brand New", "tags": ["Strategy"], "genres": [], "released": "2025-05-01", "positive": 100}}
    prompts = []
    def fake(prompt, schema):
        prompts.append(prompt)
        return {"reasons": [{"app_id": 2, "reason": "You liked A and Brand New"}]}
    out = web.reasons(lib, store, FakeRanker(), Catalog({1: "A", 2: "B", 3: "Tool"}, {1: ["Strategy"]}, {}), generate=fake)
    assert out == [{"app_id": 2, "reason": "You liked A and Brand New"}]
    assert "Brand New" in prompts[0]  # a 2025 game the 2022 catalogue does not know still has a name in the prompt


def test_reasons_explain_the_cards_the_sliders_left_on_the_page():
    lib = SteamLibrary("765", {1: 50.0}, titles={1: "A"})
    asked = []
    def fake(prompt, schema):
        asked.append(schema["properties"]["reasons"]["items"]["properties"]["app_id"]["enum"])
        return {"reasons": []}
    web.reasons(lib, STORE, FakeRanker(), Catalog({1: "A", 2: "B"}, {}, {}), generate=fake, apps=[2, 1, 99, 2])
    assert asked == [[2]]  # owned (1), unknown (99) and repeats dropped
