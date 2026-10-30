from recs.describe import Described


def test_a_new_game_is_found_by_what_it_is_about():
    # games 1 and 3 are about the same thing; 2 is about something else; 3 is new (no players at all)
    vectors = {1: [1.0, 0.0], 2: [0.0, 1.0], 3: [0.9, 0.1]}
    d = Described(vectors, fallback=[2, 1, 3])
    assert d.recommend({1: 50.0}, 2) == [3, 2]
    assert d.recommend({99: 5.0}, 2) == [2, 1]  # nothing described: the fallback


def test_popularity_can_weigh_in():
    vectors = {1: [1.0, 0.0], 2: [0.95, 0.05], 3: [0.9, 0.1]}
    d = Described(vectors, prior={2: 10, 3: 100000}, beta=1.0)
    assert d.recommend({1: 50.0}, 1) == [3]
