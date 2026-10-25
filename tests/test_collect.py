import json

from recs import collect


def page(reviews, cursor):
    return {"success": 1, "cursor": cursor, "reviews": reviews}


def review(steamid, rid, minutes=600, up=True, ts=1700000000):
    return {"recommendationid": str(rid), "voted_up": up, "timestamp_created": ts,
            "author": {"steamid": steamid, "playtime_forever": minutes, "num_reviews": 3}}


def test_pages_until_the_limit_and_stops_on_a_repeated_cursor():
    pages = iter([page([review("765611", 1), review("765612", 2)], "c2"),
                  page([review("765613", 3)], "c2"), page([], "c2")])
    got = list(collect.fetch_reviews(10, limit=5, get=lambda url: next(pages), wait=0))
    assert [r["recommendationid"] for r in got] == ["1", "2", "3"]


def test_rows_are_anonymous_and_shaped_like_the_kaggle_data():
    row = collect.to_row(553850, review("76561198000000001", 7, minutes=90, up=False), key=b"k")
    assert row == {"app_id": 553850, "date": "2023-11-14", "is_recommended": False, "hours": 1.5,
                   "user_id": collect.anonymize("76561198000000001", b"k"), "review_id": 7}
    assert "76561198000000001" not in json.dumps(row)


def test_anonymize_is_stable_per_key_and_differs_across_keys():
    a = collect.anonymize("765611", b"k1")
    assert a == collect.anonymize("765611", b"k1") and a != collect.anonymize("765611", b"k2")
    assert isinstance(a, int) and a > 0


def test_new_releases_are_sampled_at_random_not_by_fame():
    store = {a: {"released": "2026-03-01", "positive": 30 + a, "negative": 0, "genres": ["Action"]} for a in range(1, 41)}
    store[99] = {"released": "2020-01-01", "positive": 10**6, "negative": 0, "genres": ["Action"]}  # old: never
    store[98] = {"released": "2026-03-01", "positive": 3, "negative": 1, "genres": ["Action"]}  # too few reviews
    picked = collect.sample_new(store, since="2025-10-01", n=10, seed=1)
    assert len(picked) == 10 and not {98, 99} & set(picked)
    assert picked != sorted(store, key=lambda a: -store[a]["positive"])[:10]  # not simply the most reviewed
