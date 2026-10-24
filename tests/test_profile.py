from recs import profile

STORE = {
    1: {"title": "Grand Strategy A", "tags": ["Grand Strategy", "Strategy", "Multiplayer"], "genres": ["Strategy"], "released": "2016-01-01", "positive": 50000},
    2: {"title": "Shooter B", "tags": ["FPS", "Multiplayer"], "genres": ["Action"], "released": "2012-01-01", "positive": 900000},
    3: {"title": "Party C", "tags": ["Party", "Multiplayer"], "genres": ["Casual"], "released": "2018-01-01", "positive": 40000},
    4: {"title": "Wallpaper Tool", "tags": ["Utilities"], "genres": ["Utilities"], "released": "2018-01-01", "positive": 800000},
    5: {"title": "New Strategy", "tags": ["Grand Strategy", "Strategy"], "genres": ["Strategy"], "released": "2025-03-01", "positive": 30000},
    6: {"title": "New Shooter", "tags": ["FPS", "Multiplayer"], "genres": ["Action"], "released": "2025-06-01", "positive": 20000},
}


def test_taste_profile_lets_rare_tags_beat_common_ones():
    taste = profile.taste_profile({1: 900.0, 2: 20.0, 3: 20.0}, STORE, top=3)
    assert taste[0][0] == "Grand Strategy"
    assert "Multiplayer" not in [t for t, _ in taste[:2]]
    assert abs(sum(share for _, share in profile.taste_profile({1: 900.0, 2: 20.0}, STORE, top=50)) - 1) < 1e-9


def test_non_games_are_detected_by_genre():
    assert not profile.is_game(4, STORE) and profile.is_game(1, STORE)
    assert profile.is_game(999, STORE)  # unknown to the store list: keep it


def test_new_releases_match_taste_and_skip_owned_and_old():
    picks = profile.new_releases({1: 900.0, 5: 1.0}, STORE, since="2024-01-01", k=2, min_positive=1000)
    assert picks == [6] or picks[0] != 5
    assert profile.new_releases({1: 900.0}, STORE, since="2024-01-01", k=1, min_positive=1000) == [5]


def test_a_few_long_played_games_outweigh_many_short_ones():
    store = {1: {"tags": ["Grand Strategy"], "genres": []}, **{i: {"tags": ["Survival"], "genres": []} for i in range(2, 32)},
             **{i: {"tags": ["Puzzle"], "genres": []} for i in range(100, 120)}}
    library = {1: 2000.0, **{i: 5.0 for i in range(2, 32)}}
    assert profile.taste_profile(library, store, top=1)[0][0] == "Grand Strategy"


def test_format_tags_are_kept_out_of_the_taste_and_reported_as_a_split():
    store = {1: {"tags": ["Grand Strategy", "Singleplayer"], "genres": []},
             2: {"tags": ["FPS", "Multiplayer", "Free to Play"], "genres": []}, 3: {"tags": ["Puzzle"], "genres": []}}
    taste = [t for t, _ in profile.taste_profile({1: 400.0, 2: 100.0}, store, top=10)]
    assert "Singleplayer" not in taste and "Multiplayer" not in taste and "Free to Play" not in taste
    split = profile.social_split({1: 400.0, 2: 100.0, 3: 100.0}, store)
    assert split == {"multiplayer": 100 / 600, "singleplayer": 500 / 600}


def test_tag_edges_link_tags_that_share_games():
    store = {1: {"tags": ["Grand Strategy", "Historical"], "genres": [], "positive": 900},
             2: {"tags": ["Grand Strategy", "Historical"], "genres": [], "positive": 900},
             3: {"tags": ["FPS", "Shooter"], "genres": [], "positive": 900},
             4: {"tags": ["Historical", "FPS"], "genres": [], "positive": 10}}  # too few reviews to count
    edges = profile.tag_edges(["Grand Strategy", "Historical", "FPS", "Shooter"], store, min_positive=100, min_jaccard=0.3)
    assert ("Grand Strategy", "Historical", 1.0) in edges and ("FPS", "Shooter", 1.0) in edges
    assert not any({a, b} == {"Historical", "FPS"} for a, b, _ in edges)
