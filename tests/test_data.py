import json

import duckdb
import pytest

from recs import data


def write_raw(folder, reviews, games=None, tags=None):
    """reviews: (user, app, recommended, hours[, date]); games default to every reviewed app."""
    folder.mkdir(parents=True, exist_ok=True)
    rows = ["app_id,helpful,funny,date,is_recommended,hours,user_id,review_id"]
    for i, (user, app, rec, hours, *date) in enumerate(reviews):
        rows.append(f"{app},0,0,{date[0] if date else '2021-05-01'},{str(rec).lower()},{hours},{user},{i}")
    (folder / "recommendations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    apps = games if games is not None else sorted({r[1] for r in reviews})
    lines = ["app_id,title,date_release,win,mac,linux,rating,positive_ratio,user_reviews,price_final,"
             "price_original,discount,steam_deck"]
    lines += [f"{a},Game {a},2020-01-01,true,false,false,Positive,90,10,9.99,9.99,0.0,true" for a in apps]
    (folder / "games.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    tags = tags or {}
    (folder / "games_metadata.json").write_text(
        "\n".join(json.dumps({"app_id": a, "description": "", "tags": tags.get(a, ["Action"])}) for a in apps) + "\n",
        encoding="utf-8")


def positives(build_dir):
    return sorted(duckdb.sql(f"select user_id, app_id, hours from '{build_dir}/positives.parquet'").fetchall())


def test_positive_means_recommended_and_two_hours(tmp_path):
    write_raw(tmp_path / "raw", [(1, 10, True, 1.5), (1, 20, False, 9), (1, 30, True, 2.0)])
    data.build(tmp_path / "raw", tmp_path / "build", min_user=1, min_app=1)
    assert positives(tmp_path / "build") == [(1, 30, 2.0)]


def test_duplicate_reviews_keep_max_hours(tmp_path):
    write_raw(tmp_path / "raw", [(1, 10, True, 3), (1, 10, True, 7)])
    data.build(tmp_path / "raw", tmp_path / "build", min_user=1, min_app=1)
    assert positives(tmp_path / "build") == [(1, 10, 7.0)]


def test_filters_apply_twice(tmp_path):
    # user 3 survives the first user pass (2 games), loses game 30 in the app pass, then must go in the second
    reviews = [(1, 10, True, 5), (1, 20, True, 5), (2, 10, True, 5), (2, 20, True, 5), (3, 10, True, 5), (3, 30, True, 5)]
    write_raw(tmp_path / "raw", reviews)
    stats = data.build(tmp_path / "raw", tmp_path / "build", min_user=2, min_app=2)
    assert {u for u, _, _ in positives(tmp_path / "build")} == {1, 2}
    assert stats == {"users": 2, "apps": 2, "positives": 4}


def test_games_table_keeps_tags_of_surviving_apps(tmp_path):
    write_raw(tmp_path / "raw", [(1, 10, True, 5)], tags={10: ["Co-op", "Shooter"]})
    data.build(tmp_path / "raw", tmp_path / "build", min_user=1, min_app=1)
    rows = duckdb.sql(f"select app_id, title, tags from '{tmp_path}/build/games.parquet'").fetchall()
    assert rows == [(10, "Game 10", ["Co-op", "Shooter"])]


def test_build_interactions_leaves_out_users_and_apps():
    import numpy as np
    users, apps, hours = np.array([1, 1, 2, 3]), np.array([10, 20, 10, 30]), np.array([5.0, 2.0, 7.0, 3.0])
    inter = data.build_interactions(users, apps, hours, exclude_users={3}, exclude_apps=frozenset({20}))
    assert inter.item_ids.tolist() == [10] and inter.item_index == {10: 0}
    assert inter.matrix.shape == (2, 1) and inter.matrix.toarray().tolist() == [[5.0], [7.0]]


def test_quality_check_reports_the_share_of_untagged_games(tmp_path):
    write_raw(tmp_path / "raw", [(1, 10, True, 5), (1, 20, True, 5)], tags={10: ["Co-op"], 20: []})
    assert data.check_quality(data.connect(tmp_path / "raw")) == ["50.0% of games have no tags"]


def test_quality_check_rejects_negative_hours(tmp_path):
    write_raw(tmp_path / "raw", [(1, 10, True, -3)])
    with pytest.raises(data.DataQualityError, match="negative"):
        data.check_quality(data.connect(tmp_path / "raw"))


def test_quality_check_rejects_apps_missing_from_games(tmp_path):
    write_raw(tmp_path / "raw", [(1, 10, True, 5), (1, 99, True, 5)], games=[10])
    with pytest.raises(data.DataQualityError, match="missing"):
        data.check_quality(data.connect(tmp_path / "raw"))


STORE_HEADER = ("AppID,Name,Release date,Estimated owners,Peak CCU,Required age,Price,DiscountDLC count,About the game,"
                "Supported languages,Full audio languages,Reviews,Header image,Website,Support url,Support email,Windows,"
                "Mac,Linux,Metacritic score,Metacritic url,User score,Positive,Negative,Score rank,Achievements,"
                "Recommendations,Notes,Average playtime forever,Average playtime two weeks,Median playtime forever,"
                "Median playtime two weeks,Developers,Publishers,Categories,Genres,Tags,Screenshots,Movies")


def store_row(app, name, date, positive, negative, tags):
    # 40 values for 39 header names: the real file's header merges "Discount" and "DLC count"
    vals = [app, name, date, "0 - 20000", 5, 0, 9.99, 0, 0, "about", "[]", "[]", "", "", "", "", "", "True", "False",
            "False", 0, "", 0, positive, negative, "", 0, 0, "", 0, 0, 0, 0, "dev", "pub", "Co-op", "Action", tags, "", ""]
    return ",".join(f'"{v}"' if isinstance(v, str) else str(v) for v in vals)


def test_store_catalog_splits_the_merged_header(tmp_path):
    import zipfile
    zpath = tmp_path / "store.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("games.csv", "\n".join([STORE_HEADER, store_row(553850, "HELLDIVERS 2", "Feb 8, 2024", 775883, 241752,
                                                                      "Online Co-Op,PvE,Shooter")]) + "\n")
    stats = data.build_store(zpath, tmp_path / "build")
    assert stats == {"apps": 1}
    store = data.load_store(tmp_path / "build")
    assert store[553850] == {"title": "HELLDIVERS 2", "released": "2024-02-08", "positive": 775883, "negative": 241752,
                             "tags": ["Online Co-Op", "PvE", "Shooter"]}


def test_store_catalog_refuses_a_header_it_does_not_recognise(tmp_path):
    import zipfile
    zpath = tmp_path / "store.zip"
    with zipfile.ZipFile(zpath, "w") as z:
        z.writestr("games.csv", "AppID,Name\n1,x\n")
    with pytest.raises(data.DataQualityError):
        data.build_store(zpath, tmp_path / "build")
