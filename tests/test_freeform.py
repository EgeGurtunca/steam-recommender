from recs.data import Catalog
from recs.freeform import FreeFormLLM, match, normalize

CAT = Catalog({548430: "Deep Rock Galactic", 13500: "Prince of Persia: Warrior Within™", 7: "Portal 2"},
              {548430: ["Co-op"], 13500: ["Action"], 7: ["Puzzle"]}, {548430: "2020-05-13", 13500: "2008-11-21", 7: "2011-04-18"})


def test_normalize_handles_marks_and_case():
    assert normalize("Prince of Persia: Warrior Within™") == normalize("prince of persia  warrior within")


def test_match_returns_none_for_unknown():
    assert match(["Deep Rock Galactic", "Totally Real Game 9"], CAT) == [548430, None]


def test_match_tolerates_small_differences():
    assert match(["Portal II", "Prince Of Persia Warrior Within"], CAT)[1] == 13500


def test_recommend_keeps_only_matched_unowned_games():
    fake = lambda prompt, sch: {"titles": ["Portal 2", "Deep Rock Galactic", "Made Up Quest"]}
    ff = FreeFormLLM(CAT, generate=fake)
    assert ff.recommend({548430: 30.0}, k=3) == [7]
    assert ff.last == {"titles": 3, "matched": 2, "owned": 1, "on_steam": 2}


def test_titles_outside_the_catalog_can_still_be_real_steam_games():
    fake = lambda prompt, sch: {"titles": ["Old Obscure Game", "Made Up Quest"]}
    ff = FreeFormLLM(CAT, generate=fake, all_titles={**CAT.titles, 42: "Old Obscure Game"})
    assert ff.recommend({7: 3.0}, k=2) == []
    assert ff.last == {"titles": 2, "matched": 0, "owned": 0, "on_steam": 1}
