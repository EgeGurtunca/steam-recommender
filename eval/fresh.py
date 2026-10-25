"""New games through co-likes: does adding the collected 2025-2026 reviews (recs/collect.py, authors hashed) to
EASE's training find 2023+ games better than the page's tags x popularity?

The warm start eval said a week of reviews beats tags; this checks it on games the 2022 data never saw. Collected
reviewers who liked at least one 2023+ game and at least two games EASE knows are held out; their 2023+ games are
hidden, and each system ranks the same games: the 2023+ ones that appear in the collected reviews (the only
ones EASE can learn), as the page's "New since 2024" ranks only new games:

- tags x popularity: recs.profile.new_releases, what the page uses now
- popularity: the store's positive review count
- EASE trained on the 2022 positives plus every other collected reviewer, so the new games get columns

No setting is tuned here (EASE keeps lambda 500), so val and test are both reported.

The first collection took SteamSpy's top lists, so its new games are all hits; `recs.collect --new-since` adds
a random sample of new releases. Results are split by which of the two a hidden game came from, since
popularity alone finds nearly every hit and none of the rest.

Run: python -m eval.fresh
"""
import json
from collections import defaultdict

import numpy as np

from eval import metrics
from eval.intervals import per_user_recall
from recs import profile
from recs.collect import OUT
from recs.data import build_interactions, load_positives, load_store
from recs.ease import EASE

SINCE = "2023-01-01"
POWERS = (2.0, 3.0, 5.0)  # popularity's weight in the page's new-games ranking (1 = the page now)


def collected_positives() -> dict[int, dict[int, float]]:
    """Hashed author -> {app: hours} for reviews that recommend after 2 hours, as the 2022 positives are built."""
    out = defaultdict(dict)
    files = [f for f in (OUT / "reviews.jsonl", OUT / "reviews_new.jsonl") if f.exists()]  # top lists, new releases
    for line in (line for f in files for line in f.open(encoding="utf-8")):
        r = json.loads(line)
        if r["is_recommended"] and r["hours"] >= 2:
            out[r["user_id"]][r["app_id"]] = max(r["hours"], out[r["user_id"]].get(r["app_id"], 0.0))
    return out


def pick(by_user: dict, new: set[int], known: set[int], n: int, seed: int = 42) -> dict[str, list[dict]]:
    """Reviewers with a liked 2023+ game and two liked games EASE knows; half val, half test."""
    eligible = sorted(u for u, lib in by_user.items() if len(lib.keys() & new) >= 1 and len(lib.keys() & known) >= 2)
    chosen = np.random.default_rng(seed).choice(eligible, size=min(2 * n, len(eligible)), replace=False).tolist()
    entries = [{"user": u, "visible": {a: h for a, h in by_user[u].items() if a not in new},
                "hidden": sorted(by_user[u].keys() & new)} for u in chosen]
    return {"val": entries[: len(entries) // 2], "test": entries[len(entries) // 2:]}


def main() -> None:
    store = load_store()
    users, apps, hours = load_positives()
    by_user = collected_positives()
    known = set(np.unique(apps).tolist())
    new = {a for a, v in store.items() if (v.get("released") or "") >= SINCE and a not in known}
    split = pick(by_user, new, known, 1000)
    top_list = {json.loads(line)["app_id"] for line in (OUT / "reviews.jsonl").open(encoding="utf-8")}  # the hits
    held = {e["user"] for part in split.values() for e in part}
    extra = [(u, a, h) for u, lib in by_user.items() if u not in held for a, h in lib.items()]
    eu, ea, eh = (np.array(x) for x in zip(*extra))
    print(f"{len(by_user)} collected reviewers with a positive, {len(held)} held out, {len(eu)} positives added; "
          f"{len(new & set(ea.tolist()))} of the 2023+ games appear in them", flush=True)
    train = build_interactions(np.concatenate([users, eu]), np.concatenate([apps, ea]),
                               np.concatenate([hours, eh]), exclude_users=set())
    ease = EASE(train)
    pool = [a for a in train.item_ids.tolist() if a in new]
    pool_cols = np.array([train.item_index[a] for a in pool])
    weights = profile.idf(store)
    in_pool = set(pool)  # every system ranks the same games: the 2023+ ones the collected reviews cover
    by_pop = sorted(pool, key=lambda a: -store[a]["positive"])

    def ease_new(lib):
        s = ease.scores(lib)
        if s is None:
            return []
        order = np.argsort(-s[pool_cols])
        return [pool[i] for i in order if pool[i] not in lib][:10]  # top 10 of the pool

    pool_store = {a: store[a] for a in pool}

    def tags(lib, power=1.0):  # the page's ranking over the pool; the library's own games give its tag profile
        local = pool_store | {a: store[a] for a in lib if a in store}
        return [a for a in profile.new_releases(lib, local, since=SINCE, k=len(local), min_positive=0,
                                                weights=weights, pop_power=power) if a in in_pool][:10]

    systems = {"tags x popularity (the page now)": tags,
               "popularity": lambda lib: [a for a in by_pop if a not in lib][:10],
               "EASE + collected reviews": ease_new,
               **{f"tags x popularity ** {p:g}": (lambda lib, p=p: tags(lib, p)) for p in POWERS}}
    for part, entries in split.items():
        rows = [{"system": name, "per_user": [{"user": e["user"], "recs": f(e["visible"]), "hidden": e["hidden"]}
                                              for e in entries]} for name, f in systems.items()]
        base = per_user_recall(rows[0])
        print(f"\n{part}, {len(entries)} reviewers, hidden 2023+ games: {sum(len(e['hidden']) for e in entries)}")
        print("| system | recall@10 | gap vs tags x popularity, 95% interval |\n|---|---|---|")
        for r in rows:
            o = per_user_recall(r)
            gap, lo, hi = metrics.paired_bootstrap([base[u] for u in base], [o[u] for u in base])
            print(f"| {r['system']} | {np.mean(list(o.values())):.3f} | {gap:+.3f} ({lo:+.3f} to {hi:+.3f}) |")
        print("share of hidden games found, by where the game was collected:")
        for r in rows:
            found = {True: [], False: []}
            for u in r["per_user"]:
                for a in u["hidden"]:
                    found[a in top_list].append(a in u["recs"])
            print(f"  {r['system']}: top lists {np.mean(found[True]):.3f} ({len(found[True])}), "
                  f"random new releases {np.mean(found[False]):.3f} ({len(found[False])})")
            if r is rows[0]:
                base_found = found
            else:  # paired over the same hidden games
                for side, label in ((True, "top lists"), (False, "random new releases")):
                    gap, lo, hi = metrics.paired_bootstrap(base_found[side], found[side])
                    print(f"    vs tags x popularity on {label}: {gap:+.3f} ({lo:+.3f} to {hi:+.3f})")


if __name__ == "__main__":
    main()
