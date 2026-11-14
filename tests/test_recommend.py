import sys

import pytest

from recs import recommend

SPLIT = {"test": [{"user": 7, "visible": {10: 3.0, 20: 1.0}}], "val": [{"user": 9, "visible": {30: 2.0}}]}


def test_find_user_looks_in_every_group():
    assert recommend.find_user(SPLIT, 9) == {30: 2.0}
    assert recommend.find_user(SPLIT, 7) == {10: 3.0, 20: 1.0}


def test_unknown_user_is_none_not_stopiteration():
    assert recommend.find_user(SPLIT, 8) is None


@pytest.mark.parametrize("argv", [[], ["--user", "8"]])
def test_cli_exits_with_a_message(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["recommend", *argv])
    monkeypatch.setattr(recommend, "load_split", lambda: SPLIT)
    with pytest.raises(SystemExit) as e:
        recommend.main()
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert ("one of the arguments --user --library is required" in err) or ("not in the frozen eval groups" in err)
