import pytest

from recs import steam


def test_parse_profile_links():
    assert steam.parse_profile("https://steamcommunity.com/id/flop44") == ("vanity", "flop44")
    assert steam.parse_profile("steamcommunity.com/id/flop44/") == ("vanity", "flop44")
    assert steam.parse_profile("https://steamcommunity.com/profiles/76561198000000001/games") == ("id", "76561198000000001")
    assert steam.parse_profile("76561198000000001") == ("id", "76561198000000001")
    with pytest.raises(ValueError):
        steam.parse_profile("https://example.com/user/1")


def test_library_from_a_vanity_link():
    answers = {
        "ResolveVanityURL": {"response": {"success": 1, "steamid": "76561198000000001"}},
        "GetOwnedGames": {"response": {"game_count": 2, "games": [
            {"appid": 553850, "name": "HELLDIVERS 2", "playtime_forever": 6000, "playtime_2weeks": 120},
            {"appid": 620, "name": "Portal 2", "playtime_forever": 0}]}},
    }
    get = lambda url: next(v for k, v in answers.items() if k in url)
    lib = steam.load_library("https://steamcommunity.com/id/flop44", key="k", get=get)
    assert lib.games == {553850: 100.0, 620: 0.0}
    assert lib.recent == {553850: 2.0}
    assert lib.titles[620] == "Portal 2"


def test_private_library_is_a_clear_error():
    get = lambda url: {"response": {}}
    with pytest.raises(steam.PrivateLibrary):
        steam.load_library("76561198000000001", key="k", get=get)
