"""Warm start: how much early data does a new game need before EASE finds it?

The cold-start run takes the 626 games released from July 2022 out of training entirely. Real stores do not
wait for that: a game's first players start linking it to the rest of the catalogue on day one (a patent on
game recommendations adds a co-play graph for exactly this). Here each new game gets back its early positive
reviews, from training users only (never an eval user), EASE is retrained, and it has to find the hidden new
games of the same users as the cold-start run. "Early" is either the first k reviews (--by count) or the
reviews from the first k days after release (--by days, what a store actually has a week or a month in).

Run: python -m eval.warm_start --by days --ks 7,30,90,365
     python -m eval.warm_start --by count --ks 0,5,20,50,200
"""
import argparse

import numpy as np

from eval import metrics
from eval.run_eval import NEW_FROM, evaluate, table
from recs.data import build_interactions, connect, load_catalog, load_positives
from recs.ease import EASE
from recs.popularity import Popularity
from recs.split import load_split


def early_reviews(new_apps: set[int], users: np.ndarray, by: str = "count") -> tuple[np.ndarray, ...]:
    """(user, app, hours, position) for every positive review of a new game by a training user. The position is
    the review's order (counting all reviewers, eval users too: "first 20" means the game's first 20 reviews, of
    which training keeps its own) or its day after release (a review before release counts as day 0)."""
    position = ("row_number() over (partition by app_id order by date, review_id)" if by == "count"
                else "greatest(date_diff('day', date_release, date), 0)")
    con = connect()
    con.execute("create temp table new_apps (app_id bigint)")
    con.executemany("insert into new_apps values (?)", [(a,) for a in new_apps])
    cols = con.execute(f"""
        select user_id, app_id, hours, {position} as n
        from (select user_id, app_id, max(hours) as hours, min(date) as date, min(review_id) as review_id
              from recommendations where is_recommended and hours >= 2 group by user_id, app_id)
        join games using (app_id)
        where app_id in (select app_id from new_apps)""").fetchnumpy()
    keep = np.isin(cols["user_id"], users)
    return cols["user_id"][keep], cols["app_id"][keep], cols["hours"][keep], cols["n"][keep]


def with_first(k: int, base: tuple, early: tuple, eval_users: set[int]):
    keep = early[3] <= k
    return build_interactions(*(np.concatenate([b, e[keep]]) for b, e in zip(base, early[:3])),
                              exclude_users=eval_users)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--by", choices=["count", "days"], default="days")
    ap.add_argument("--ks", default="7,30,90,365")
    args = ap.parse_args()
    split = load_split()
    eval_users = {e["user"] for part in split.values() for e in part}
    catalog = load_catalog()
    new_apps = {a for a, d in catalog.released.items() if d >= NEW_FROM}
    users, apps, hours = load_positives()
    old = ~np.isin(apps, list(new_apps))
    base = (users[old], apps[old], hours[old])
    early = early_reviews(new_apps, np.setdiff1d(np.unique(base[0]), list(eval_users)), args.by)
    print(f"{len(new_apps)} new games, {len(early[0])} positive reviews of them by training users")
    rows = {"val": [], "test": []}
    ks = [int(k) for k in args.ks.split(",")]
    for i, k in enumerate(ks, 1):
        train = with_first(k, base, early, eval_users)
        pop = Popularity(train)
        system = EASE(train, fallback=pop.ranking)
        system.name = f"ease, first {k} {'reviews' if args.by == 'count' else 'days'}"
        pop_rank = {app: i + 1 for i, app in enumerate(pop.ranking)}
        for part in rows:
            rows[part].append(evaluate(system, metrics.cold_start_entries(split[part], new_apps), pop_rank))
        print(f"[{i}/{len(ks)}] k={k}: val {rows['val'][-1]['recall@10']:.3f}, "
              f"test {rows['test'][-1]['recall@10']:.3f}", flush=True)
    for part, r in rows.items():
        print(f"\n{part}, {len(r[0]['per_user'])} users with a hidden new game")
        print(table(r))


if __name__ == "__main__":
    main()
