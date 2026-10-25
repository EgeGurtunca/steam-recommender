"""A fresh, anonymous review dataset for 2023-2026, built the way the Kaggle one was: public store reviews only.

Each review's author id is turned into a keyed hash (HMAC-SHA256 with a key that never leaves this machine)
the moment it arrives; the Steam id itself is never written anywhere. No profiles or libraries are read.
Requests are spaced at least a second apart.

Run: python -m recs.collect --apps 400 --per-app 1000
     python -m recs.collect --new-since 2025-10-25 --apps 400 --per-app 300   (new releases, sampled at random)
"""
import argparse
import hashlib
import hmac
import json
import secrets
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator

from recs.data import ROOT

OUT = ROOT / "data" / "collected"
KEY_FILE = OUT / "hash.key"
# The store's /appreviews JSON was switched off on 2026-10-22; this Web API method replaced it (no key needed).
REVIEWS = "https://api.steampowered.com/IUserReviewsService/GetAppReviews/v1/?input_json={query}"
DAYS = 365  # the method looks back at most a year: newest first, all languages
STEAMSPY = "https://steamspy.com/api.php?request={req}"
UA = {"User-Agent": "steam-recommender research crawler (public reviews only, 1 req/s)"}


def http_json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
        return json.load(r)


def anonymize(steamid: str, key: bytes) -> int:
    """Stable per key, not reversible without it; 63 bits so it fits the BIGINT user_id column."""
    return int.from_bytes(hmac.new(key, steamid.encode(), hashlib.sha256).digest()[:8], "big") >> 1


def to_row(app: int, review: dict, key: bytes) -> dict:
    return {"app_id": app,
            "date": datetime.fromtimestamp(review["timestamp_created"], timezone.utc).date().isoformat(),
            "is_recommended": bool(review["voted_up"]),
            "hours": round(review["author"]["playtime_forever"] / 60, 1),
            "user_id": anonymize(review["author"]["steamid"], key),
            "review_id": int(review["recommendationid"])}


def fetch_reviews(app: int, limit: int, get: Callable[[str], dict] = http_json, wait: float = 1.0) -> Iterator[dict]:
    """Newest reviews first, 100 per request, until `limit`, the end, or a cursor that stops moving."""
    cursor, seen, count = "*", set(), 0
    while count < limit:
        query = {"appid": app, "num_per_page": 100, "filter_type": "recent", "languages": ["all"], "day_range": DAYS,
                 "cursor": cursor}
        data = get(REVIEWS.format(query=urllib.parse.quote(json.dumps(query))))
        data = data.get("response", data)
        batch = data.get("reviews") or []
        for r in batch[: limit - count]:
            yield r
        count += len(batch)
        cursor = data.get("cursor", "")
        if not batch or cursor in seen:
            return
        seen.add(cursor)
        time.sleep(wait)


def load_key() -> bytes:
    OUT.mkdir(parents=True, exist_ok=True)
    if not KEY_FILE.exists():
        KEY_FILE.write_bytes(secrets.token_bytes(32))
    return KEY_FILE.read_bytes()


def pick_apps(n: int, old: list[int], get: Callable[[str], dict] = http_json) -> list[int]:
    """SteamSpy's public top lists (where the 2023-2026 games are), then the dataset's most popular games."""
    apps = []
    for req in ("top100in2weeks", "top100forever"):
        apps += [int(a) for a in get(STEAMSPY.format(req=req))]
        time.sleep(1)
    for a in old:
        if len(apps) >= n * 2:
            break
        apps.append(a)
    return list(dict.fromkeys(apps))[:n]


def sample_new(store: dict, since: str, n: int, min_reviews: int = 20, seed: int = 0) -> list[int]:
    """A random sample of games released from `since` with at least `min_reviews` reviews: new releases as they
    come, mostly not hits, which the top lists above never reach."""
    import random
    from recs.profile import is_game
    pool = sorted(a for a, v in store.items() if (v.get("released") or "") >= since and is_game(a, store)
                  and v.get("positive", 0) + v.get("negative", 0) >= min_reviews)
    return sorted(random.Random(seed).sample(pool, min(n, len(pool))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apps", type=int, default=400)
    ap.add_argument("--per-app", type=int, default=1000)
    ap.add_argument("--new-since", help="instead of the top lists: a random sample of games released from this date "
                                        "(store catalogue), written to reviews_new.jsonl")
    args = ap.parse_args()
    from recs.data import build_interactions, load_positives, load_store
    from recs.popularity import Popularity
    key, out = load_key(), OUT / ("reviews_new.jsonl" if args.new_since else "reviews.jsonl")
    done = {json.loads(line)["app_id"] for line in out.open(encoding="utf-8")} if out.exists() else set()
    if args.new_since:
        apps = [a for a in sample_new(load_store(), args.new_since, args.apps) if a not in done]
    else:
        popular = Popularity(build_interactions(*load_positives(), exclude_users=set())).ranking[:300]
        apps = [a for a in pick_apps(args.apps, popular) if a not in done]
    with out.open("a", encoding="utf-8") as f:
        for i, app in enumerate(apps, 1):
            try:
                rows = [to_row(app, r, key) for r in fetch_reviews(app, args.per_app)]
            except Exception as e:  # one game failing should not lose the rest
                print(f"app {app}: {type(e).__name__}: {e}", flush=True)
                continue
            f.writelines(json.dumps(r) + "\n" for r in rows)
            f.flush()
            if i % 20 == 0:
                print(f"PROGRESS {i}/{len(apps)} apps", flush=True)
    print(f"done: {out}")


if __name__ == "__main__":
    main()
