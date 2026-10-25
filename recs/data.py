"""From the Kaggle "Game Recommendations on Steam" zip to two parquet files, in SQL.

positives.parquet: one row per (user, app) the user recommended after at least `min_hours` of play, users and
apps with too few of those filtered out. games.parquet: title, release date and tags of the apps that survive.

Run: python -m recs.data <path to steam_game_recommendations_archive.zip>
     python -m recs.data --store <path to the 2026 store catalogue zip>
"""
import csv
import io
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
BUILD_DIR = ROOT / "data" / "build"
FILES = ("recommendations.csv", "games.csv", "games_metadata.json", "users.csv")


class DataQualityError(Exception):
    pass


def extract(zip_path: Path, raw_dir: Path = RAW_DIR) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        for name in FILES:
            if not (raw_dir / name).exists():
                z.extract(name, raw_dir)


def connect(raw_dir: Path = RAW_DIR) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB with the raw files as views: recommendations, games, metadata."""
    con = duckdb.connect()
    con.execute(f"""
        create view recommendations as select * from read_csv('{raw_dir.as_posix()}/recommendations.csv', header = true,
            columns = {{'app_id': 'BIGINT', 'helpful': 'BIGINT', 'funny': 'BIGINT', 'date': 'DATE',
                        'is_recommended': 'BOOLEAN', 'hours': 'DOUBLE', 'user_id': 'BIGINT', 'review_id': 'BIGINT'}});
        create view games as select app_id, title, date_release
            from read_csv('{raw_dir.as_posix()}/games.csv', header = true, types = {{'app_id': 'BIGINT', 'date_release': 'DATE'}});
        create view metadata as select app_id, tags
            from read_json('{raw_dir.as_posix()}/games_metadata.json', format = 'newline_delimited',
                           columns = {{'app_id': 'BIGINT', 'description': 'VARCHAR', 'tags': 'VARCHAR[]'}});
    """)
    return con


def check_quality(con: duckdb.DuckDBPyConnection) -> list[str]:
    """Stop on data that would silently poison the model; return softer problems as warnings."""
    negative = con.execute("select count(*) from recommendations where hours < 0").fetchone()[0]
    if negative:
        raise DataQualityError(f"{negative} reviews with negative hours")
    lo, hi = con.execute("select min(date), max(date) from recommendations").fetchone()
    if str(lo) < "2010-01-01" or str(hi) > "2023-12-31":
        raise DataQualityError(f"review dates {lo}..{hi} outside 2010-01-01..2023-12-31")
    missing = con.execute("select count(distinct app_id) from recommendations anti join games using (app_id)").fetchone()[0]
    if missing:
        raise DataQualityError(f"{missing} reviewed app ids missing from games.csv")
    untagged = con.execute("select avg(case when len(coalesce(tags, [])) = 0 then 1 else 0 end) from games "
                       "left join metadata using (app_id)").fetchone()[0]
    return [f"{untagged:.1%} of games have no tags"] if untagged else []


def build(raw_dir: Path = RAW_DIR, build_dir: Path = BUILD_DIR, min_user: int = 5, min_app: int = 50,
          min_hours: float = 2.0) -> dict[str, int]:
    con = connect(raw_dir)
    con.execute("create table pos as select user_id, app_id, max(hours) as hours from recommendations "
                "where is_recommended and hours >= ? group by user_id, app_id", [min_hours])
    for _ in range(2):  # removing users can starve apps and the other way round; twice gets almost all of it
        con.execute("delete from pos where user_id in (select user_id from pos group by user_id having count(*) < ?)",
                    [min_user])
        con.execute("delete from pos where app_id in (select app_id from pos group by app_id having count(*) < ?)",
                    [min_app])
    build_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"copy (select * from pos order by user_id, app_id) to '{build_dir.as_posix()}/positives.parquet' (format parquet)")
    con.execute(f"""copy (select app_id, title, date_release, coalesce(tags, []) as tags
                          from games left join metadata using (app_id)
                          where app_id in (select app_id from pos) order by app_id)
                    to '{build_dir.as_posix()}/games.parquet' (format parquet)""")
    users, apps, rows = con.execute("select count(distinct user_id), count(distinct app_id), count(*) from pos").fetchone()
    return {"users": users, "apps": apps, "positives": rows}


@dataclass
class Catalog:
    titles: dict[int, str]
    tags: dict[int, list[str]]
    released: dict[int, str]  # ISO date


def load_catalog(build_dir: Path = BUILD_DIR) -> Catalog:
    rows = duckdb.execute(f"select app_id, title, date_release, tags from '{build_dir.as_posix()}/games.parquet'").fetchall()
    return Catalog({a: t for a, t, _, _ in rows}, {a: list(g) for a, _, _, g in rows}, {a: str(d) for a, _, d, _ in rows})


def load_all_titles(raw_dir: Path = RAW_DIR) -> dict[int, str]:
    """Every app in games.csv, filtered out or not (the free-form experiment checks LLM titles against it)."""
    return dict(duckdb.execute(f"select app_id, title from read_csv('{raw_dir.as_posix()}/games.csv', header = true)").fetchall())


def load_positives(build_dir: Path = BUILD_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(users, apps, hours), sorted by user then app."""
    cols = duckdb.execute(f"select user_id, app_id, hours from '{build_dir.as_posix()}/positives.parquet' "
                          "order by user_id, app_id").fetchnumpy()
    return cols["user_id"], cols["app_id"], cols["hours"]


def load_positive_dates(build_dir: Path = BUILD_DIR) -> tuple[np.ndarray, ...]:
    """(users, apps, hours, dates) for every positive, the date being the first review that made it one (cached)."""
    path = build_dir / "positive_dates.parquet"
    if not path.exists():
        con = connect()
        con.execute(f"""copy (
            select p.user_id, p.app_id, p.hours, r.date
            from '{(build_dir / "positives.parquet").as_posix()}' p
            join (select user_id, app_id, min(date) as date from recommendations
                  where is_recommended and hours >= 2 group by user_id, app_id) r using (user_id, app_id)
            order by user_id, app_id) to '{path.as_posix()}' (format parquet)""")
    cols = duckdb.execute(f"select * from '{path.as_posix()}' order by user_id, app_id").fetchnumpy()
    return cols["user_id"], cols["app_id"], cols["hours"], cols["date"].astype("datetime64[D]")


@dataclass
class Interactions:
    matrix: csr_matrix  # users x items, hours as values
    item_ids: np.ndarray  # column -> app_id
    item_index: dict[int, int]  # app_id -> column


def build_interactions(users: np.ndarray, apps: np.ndarray, hours: np.ndarray, exclude_users: set[int],
                       exclude_apps: frozenset[int] = frozenset()) -> Interactions:
    """The training matrix: eval users left out entirely, and (for the cold-start test) the "new" apps too."""
    keep = ~np.isin(users, list(exclude_users)) & ~np.isin(apps, list(exclude_apps))
    u, a, h = users[keep], apps[keep], hours[keep]
    user_ids, rows = np.unique(u, return_inverse=True)
    item_ids, cols = np.unique(a, return_inverse=True)
    matrix = csr_matrix((h.astype(np.float32), (rows, cols)), shape=(len(user_ids), len(item_ids)))
    return Interactions(matrix, item_ids, {int(app): i for i, app in enumerate(item_ids)})


def build_store(zip_path: Path, build_dir: Path = BUILD_DIR) -> dict[str, int]:
    """The October 2026 store catalogue (130K apps): tags and review counts for games the 2022 data never saw.

    Its header merges two columns into "DiscountDLC count" while every row has both values, which shifts every
    later column by one (Positive reads 0, Tags holds the genres). Split it, or refuse a header we don't know.
    """
    csv.field_size_limit(10**9)
    with zipfile.ZipFile(zip_path) as z, z.open("games.csv") as f:
        rows = csv.reader(io.TextIOWrapper(f, encoding="utf-8"))
        header = next(rows)
        if "DiscountDLC count" not in header:
            raise DataQualityError(f"unexpected store catalogue header: {header[:8]}...")
        i = header.index("DiscountDLC count")
        header[i:i + 1] = ["Discount", "DLC count"]
        col = {h: n for n, h in enumerate(header)}
        out = []
        for r in rows:
            if len(r) != len(header):
                raise DataQualityError(f"row with {len(r)} values for {len(header)} columns: {r[:2]}")
            try:
                released = datetime.strptime(r[col["Release date"]], "%b %d, %Y").date().isoformat()
            except ValueError:
                released = None  # "Coming soon", "Q1 2027" and the like
            out.append((int(r[col["AppID"]]), r[col["Name"]], released, int(r[col["Positive"]] or 0),
                        int(r[col["Negative"]] or 0), [t for t in r[col["Tags"]].split(",") if t],
                        [g for g in r[col["Genres"]].split(",") if g], int(r[col["Median playtime forever"]] or 0)))
    build_dir.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute("create table store (app_id bigint, title varchar, released date, positive bigint, negative bigint, "
                "tags varchar[], genres varchar[], median_minutes bigint)")  # median 0 = SteamSpy has no number
    con.executemany("insert into store values (?, ?, ?, ?, ?, ?, ?, ?)", out)
    con.execute(f"copy store to '{build_dir.as_posix()}/store.parquet' (format parquet)")
    return {"apps": len(out)}


def load_store(build_dir: Path = BUILD_DIR) -> dict[int, dict]:
    rows = duckdb.execute(f"select * from '{build_dir.as_posix()}/store.parquet'").fetchall()
    return {a: {"title": t, "released": str(d) if d else None, "positive": p, "negative": n, "tags": list(g),
                "genres": list(ge), "median_hours": m / 60} for a, t, d, p, n, g, ge, m in rows}


if __name__ == "__main__":
    if sys.argv[1] == "--store":  # python -m recs.data --store <October 2026 store catalogue zip>
        print(build_store(Path(sys.argv[2])))
        raise SystemExit
    extract(Path(sys.argv[1]))
    for warning in check_quality(connect()):
        print("warning:", warning)
    print(build())
