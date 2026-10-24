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
