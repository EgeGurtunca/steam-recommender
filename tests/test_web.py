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


def test_demo_needs_no_steam_key_or_network(monkeypatch):
    def no_network(link):
        raise AssertionError("demo must not call Steam")
    monkeypatch.setattr(web.steam, "load_library", no_network)
    lib = web.library(" demo ")
    assert lib.steamid == "demo" and len(lib.games) >= 10 and set(lib.recent) <= set(lib.games)


def test_the_ranker_sees_the_latest_played_games_weighing_most():
    seen = {}

    class Recording(FakeRanker):
        def ranking(self, library, n):
            seen.update(library)
            return super().ranking(library, n)

    day = 86400
    lib = SteamLibrary("765", {1: 50.0, 2: 30.0, 9: 1.0}, last_played={1: 1000 * day, 2: 820 * day})
    web.next_picks(lib, STORE, Recording(), now=1000 * day)
    assert seen[1] == 1.0 and abs(seen[2] - 0.5) < 1e-9 and 9 not in seen  # 180 days older: half; under 2 h: out


def test_cards_say_which_played_game_led_to_them():
    class WithEase(FakeRanker):
        class ease:
            @staticmethod
            def because(library, picks):
                return {2: 1}

    lib = SteamLibrary("765", {1: 50.0}, titles={1: "A"})
    p = web.payload(lib, STORE, WithEase(), Catalog({1: "A", 2: "B", 3: "Tool"}, {}, {}))
    assert [(g["app_id"], g.get("because")) for g in p["next"]] == [(2, "A")]


def test_a_page_takes_well_under_a_second_on_a_store_of_real_size():
    import random
    import time
    rng = random.Random(0)
    names = [f"tag{i}" for i in range(300)]
    store = {a: {"title": f"G{a}", "tags": rng.sample(names, 12), "genres": ["Action"], "positive": rng.randint(0, 10**5),
                 "negative": 10, "released": f"{rng.randint(2005, 2025)}-01-01"} for a in range(1, 20001)}
    lib = SteamLibrary("765", {a: float(rng.randint(2, 300)) for a in range(1, 201)}, last_played={})

    class Many:
        def ranking(self, library, n):
            return list(range(201, 201 + n))

    start = time.perf_counter()
    p = web.payload(lib, store, Many(), Catalog({}, {}, {}))
    assert len(p["next"]) == web.CANDIDATES and time.perf_counter() - start < 1.0


def test_the_last_two_weeks_get_their_own_picks_and_cards_carry_a_price():
    seen = []

    class Recording(FakeRanker):
        def ranking(self, library, n):
            seen.append(set(library))
            return super().ranking(library, n)

    store = {**STORE, 1: {**STORE[1], "price": 0.0}, 2: {**STORE[2], "price": 19.99}}
    lib = SteamLibrary("765", {1: 50.0, 7: 30.0}, recent={7: 3.0})
    p = web.payload(lib, store, Recording(), Catalog({1: "A", 2: "B", 3: "Tool"}, {}, {}))
    assert seen == [{1, 7}, {7}]  # all played games, then only the last two weeks' (its hours are its total)
    assert p["next"][0]["price"] == 19.99 and [g["app_id"] for g in p["next_recent"]] == [2]
    assert web.payload(SteamLibrary("765", {1: 50.0}), store, FakeRanker(), Catalog({}, {}, {}))["next_recent"] == []


def test_play_together_picks_multiplayer_games_neither_will_hate():
    social = {"title": "", "genres": ["Action"], "released": "2020-01-01", "positive": 1000}
    store = {10: {**social, "title": "Co-op Hit", "tags": ["Online Co-Op"]},
             11: {**social, "title": "Solo Hit", "tags": ["Singleplayer"]},
             12: {**social, "title": "Only A Likes", "tags": ["Multiplayer"]},
             13: {**social, "title": "Both Own", "tags": ["Co-op"]},
             1: {**social, "title": "A1", "tags": ["RPG"]}, 2: {**social, "title": "B1", "tags": ["RPG"]}}

    class Ranks:
        def ranking(self, library, n):
            return [11, 12, 10] if 1 in library else [10, 11, 99, 98, 97]  # 12 never in B's list

    a = SteamLibrary("a", {1: 50.0, 13: 20.0})
    b = SteamLibrary("b", {2: 40.0, 13: 5.0, 10: 3.0})
    t = web.together(a, b, store, Ranks(), Catalog({}, {}, {}))
    assert [g["app_id"] for g in t["both_own"]] == [13]
    assert [(g["app_id"], g["owned_by"]) for g in t["new"]] == [(10, "friend")]  # 11 is solo, 12 B would not like
