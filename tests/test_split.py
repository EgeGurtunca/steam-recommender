import numpy as np
import pytest

from recs import split


def synthetic(n_users=60, seed=0):
    """Users 1..n with 8 to 30 positives each, sorted by user like positives.parquet."""
    rng = np.random.default_rng(seed)
    users, apps = [], []
    for u in range(1, n_users + 1):
        games = rng.choice(np.arange(100, 400), size=rng.integers(8, 31), replace=False)
        users += [u] * len(games)
        apps += sorted(games.tolist())
    return np.array(users), np.array(apps), np.full(len(users), 5.0)


def test_split_is_deterministic_and_disjoint():
    users, apps, hours = synthetic()
    a = split.make_split(users, apps, hours, n_val=5, n_test=5)
    b = split.make_split(users, apps, hours, n_val=5, n_test=5)
    assert a == b
    val, test = {e["user"] for e in a["val"]}, {e["user"] for e in a["test"]}
    assert len(val) == len(test) == 5 and not val & test
    counts = {u: int((users == u).sum()) for u in val | test}
    assert min(counts.values()) >= 10


def test_hides_twenty_percent_at_least_two():
    users = np.array([1] * 10 + [2] * 30)
    apps = np.array(list(range(10)) + list(range(100, 130)))
    s = split.make_split(users, apps, np.full(40, 3.0), n_val=1, n_test=1)
    by_user = {e["user"]: e for e in s["val"] + s["test"]}
    assert len(by_user[1]["hidden"]) == 2 and len(by_user[2]["hidden"]) == 6
    for u, entry in by_user.items():
        owned = set(apps[users == u].tolist())
        assert not set(entry["hidden"]) & set(entry["visible"])
        assert set(entry["hidden"]) | set(entry["visible"]) == owned


def test_split_file_is_not_overwritten(tmp_path):
    users, apps, hours = synthetic()
    path = tmp_path / "split.json"
    split.save_split(split.make_split(users, apps, hours, n_val=2, n_test=2), path)
    before = path.read_text()
    with pytest.raises(FileExistsError):
        split.save_split(split.make_split(users, apps, hours, n_val=3, n_test=3, seed=7), path)
    assert path.read_text() == before


def test_saved_split_loads_back_with_int_keys(tmp_path):
    users, apps, hours = synthetic()
    s = split.make_split(users, apps, hours, n_val=2, n_test=2)
    split.save_split(s, tmp_path / "split.json")
    assert split.load_split(tmp_path / "split.json") == s


def test_time_split_shows_the_past_and_hides_the_future():
    from eval.time_split import make_time_split
    users = np.array([1] * 6 + [2] * 6 + [3] * 3)
    apps = np.array([10, 11, 12, 13, 14, 15] * 2 + [10, 11, 12])
    hours = np.ones(15)
    old, new = np.datetime64("2021-06-01"), np.datetime64("2022-03-01")
    dates = np.array([old] * 4 + [new] * 2 + [old] * 6 + [old, new, new])
    split = make_time_split(users, apps, hours, dates, cutoff="2022-01-01", n_val=1, n_test=0, min_before=3, min_after=2)
    # user 2 has nothing after the cutoff, user 3 too little before it: only user 1 qualifies
    assert split["val"] == [{"user": 1, "visible": {10: 1.0, 11: 1.0, 12: 1.0, 13: 1.0}, "hidden": [14, 15]}]


def test_by_recency_weighs_each_visible_game_by_when_the_user_liked_it():
    from eval.time_split import by_recency
    entries = [{"user": 1, "visible": {10: 50.0, 11: 3.0}, "hidden": [12]}]
    liked_on = {(1, 10): np.datetime64("2021-12-31"), (1, 11): np.datetime64("2020-01-01")}
    out = by_recency(entries, liked_on, half_life=730.0, cutoff="2022-01-01")
    assert out[0]["hidden"] == [12] and entries[0]["visible"][10] == 50.0  # the original is untouched
    assert abs(out[0]["visible"][10] - 0.5 ** (1 / 730)) < 1e-9 and abs(out[0]["visible"][11] - 0.5) < 0.01


def test_transitions_count_what_each_user_liked_next():
    from recs.sequence import transitions
    users = np.array([1, 1, 1, 2, 2])
    apps = np.array([10, 11, 12, 10, 12])
    dates = np.array(["2020-01-01", "2020-02-01", "2020-03-01", "2020-01-01", "2020-05-01"], dtype="datetime64[D]")
    index = {10: 0, 11: 1, 12: 2}
    t = transitions(users, apps, dates, index, window=1).toarray()
    # user 1: 10 -> 11 -> 12, user 2: 10 -> 12; rows are normalised to sum to 1
    assert t[0].tolist() == [0, 0.5, 0.5] and t[1].tolist() == [0, 0, 1] and t[2].sum() == 0


def test_a_stale_transition_cache_is_counted_again(tmp_path, monkeypatch):
    from scipy.sparse import csr_matrix, save_npz

    from recs import data, sequence
    save_npz(tmp_path / "t.npz", csr_matrix((2, 2), dtype=np.float32))  # built for two games
    dates = np.array(["2020-01-01", "2020-02-01"], dtype="datetime64[D]")
    monkeypatch.setattr(data, "load_positive_dates", lambda: (np.array([1, 1]), np.array([10, 11]), None, dates))
    t = sequence.load_transitions({10: 0, 11: 1, 12: 2}, tmp_path / "t.npz")
    assert t.shape == (3, 3) and t[0, 1] == 1.0


def test_fresh_holds_out_reviewers_and_hides_only_their_new_games():
    from eval.fresh import pick
    by_user = {1: {10: 5.0, 11: 3.0, 99: 8.0},  # two known games and a new one: eligible
               2: {10: 5.0, 99: 2.0},  # one known game: not enough
               3: {10: 5.0, 11: 3.0}}  # nothing new: nothing to find
    split = pick(by_user, new={99}, known={10, 11}, n=1)
    assert split["val"] == [] and split["test"] == [{"user": 1, "visible": {10: 5.0, 11: 3.0}, "hidden": [99]}]


def test_latest_keeps_only_the_most_recently_liked_games():
    from eval.time_split import latest
    entries = [{"user": 1, "visible": {10: 1.0, 11: 2.0, 12: 3.0}, "hidden": [13]}]
    liked_on = {(1, 10): np.datetime64("2021-03-01"), (1, 11): np.datetime64("2019-01-01"),
                (1, 12): np.datetime64("2021-06-01")}
    assert latest(entries, liked_on, 2) == [{"user": 1, "visible": {10: 1.0, 12: 3.0}, "hidden": [13]}]
