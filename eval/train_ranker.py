"""Train the second stage (recs/ranker.py) and score it against plain EASE on the frozen split.

Ranker users are drawn like the eval users (20% of their games hidden) from the training users, and taken out of
EASE's training together with the eval users, so both systems below run on the same EASE.

Run: python -m eval.train_ranker --set val --ranker-users 20000
"""
import argparse

import numpy as np

from eval.run_eval import evaluate, table
from recs.content import Content
from recs.data import build_interactions, load_catalog, load_positives, load_store
from recs.ease import EASE
from recs.popularity import Popularity
from recs.ranker import Features, Ranker
from recs.split import load_split, make_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], default="val")
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--ranker-users", type=int, default=20000)
    args = ap.parse_args()

    split = load_split()
    eval_users = {e["user"] for part in split.values() for e in part}
    users, apps, hours = load_positives()
    rest = ~np.isin(users, list(eval_users))
    ranker_entries = make_split(users[rest], apps[rest], hours[rest], n_val=args.ranker_users, n_test=0, seed=7)["val"]
    ranker_users = {e["user"] for e in ranker_entries}
    print(f"[1/4] training EASE without {len(eval_users)} eval and {len(ranker_users)} ranker users", flush=True)
    train = build_interactions(users, apps, hours, exclude_users=eval_users | ranker_users)
    pop = Popularity(train)
    ease = EASE(train, fallback=pop.ranking)
    catalog = load_catalog()
    counts = np.asarray((train.matrix > 0).sum(axis=0)).ravel()
    features = Features(ease, Content(catalog), dict(zip(train.item_ids.tolist(), counts.tolist())), load_store(),
                        catalog.released)

    print(f"[2/4] candidate features for {len(ranker_entries)} ranker users", flush=True)
    ranker = Ranker(features, fallback=pop.ranking).fit(
        ranker_entries, progress=lambda n, total: print(f"  features {n}/{total}", flush=True))
    print(f"[3/4] trained: {ranker.model.n_iter_} trees", flush=True)

    entries = split[args.set][: args.users]
    pop_rank = {app: i + 1 for i, app in enumerate(pop.ranking)}
    rows = [evaluate(ease, entries, pop_rank), evaluate(ranker, entries, pop_rank)]
    print(f"[4/4] {args.set}, {len(entries)} users")
    print(table(rows))


if __name__ == "__main__":
    main()
