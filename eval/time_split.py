"""Evaluate as if the system had been deployed on CUTOFF: everything trains on reviews before it, and each eval
user's games liked after it are what the system has to find.

The main split (eval/split.json) hides a random 20% of each library, so a game liked in 2022 can help predict one
liked in 2015, and the training users' later reviews are in the model too. Here neither can happen. Games
released after the cutoff stay in the hidden lists although no model can know them: a deployed system would
miss them too.

Run: python -m eval.time_split --make                      (writes eval/time_split.json once, refuses to overwrite)
     python -m eval.time_split --set val --systems popularity,content,als,ease
     python -m eval.time_split --set val --systems ease,pop-recent:30,ease+recent:0.3,ease-decay:365   (time-aware)
"""
import argparse
import json
import math
from datetime import datetime

import numpy as np

from eval.metrics import recall_at_k
from eval.run_eval import RESULTS, Systems, evaluate, table
from recs.core import top_up
from recs.data import ROOT, build_interactions, load_catalog, load_positive_dates as positive_dates
from recs.ease import EASE
from recs.popularity import Popularity
from recs.sasrec_np import SasrecNudge
from recs.sequence import HALF_LIFE, Markov, MarkovNudge, transitions
from recs.split import load_split, save_split

CUTOFF = "2022-01-01"
TIME_SPLIT_PATH = ROOT / "eval" / "time_split.json"
RECENT_DAYS = 30  # "lately": the best window for recent popularity on val (30, 90, 180 tried)


def make_time_split(users: np.ndarray, apps: np.ndarray, hours: np.ndarray, dates: np.ndarray, cutoff: str = CUTOFF,
                    n_val: int = 1000, n_test: int = 1000, min_before: int = 8, min_after: int = 2,
                    seed: int = 42) -> dict:
    """users sorted. Eligible: at least min_before positives before the cutoff and min_after on or after it."""
    rng = np.random.default_rng(seed)
    after = dates >= np.datetime64(cutoff)
    ids, starts, counts = np.unique(users, return_index=True, return_counts=True)
    n_after = np.add.reduceat(after.astype(int), starts)
    eligible = np.flatnonzero((counts - n_after >= min_before) & (n_after >= min_after))
    picked = rng.choice(eligible, size=n_val + n_test, replace=False)
    out = {"val": [], "test": []}
    for n, i in enumerate(picked):
        rows = np.arange(starts[i], starts[i] + counts[i])
        old, new = rows[~after[rows]], rows[after[rows]]
        out["val" if n < n_val else "test"].append({
            "user": int(ids[i]), "visible": dict(zip(apps[old].tolist(), hours[old].tolist())),
            "hidden": sorted(apps[new].tolist())})
    return out


class RecentNudge:
    """EASE's top 100, each moved up by how many training users liked it in the last weeks before the cutoff:
    what is being played now, which a model of all past years under-weights."""

    def __init__(self, ease, recent: dict[int, int], beta: float, fallback=(), n: int = 100):
        self.ease, self.recent, self.beta, self.fallback, self.n = ease, recent, beta, fallback, n
        self.name = f"ease+recent:{beta:g}"
        self.top = math.log1p(max(recent.values(), default=0)) or 1.0

    def recommend(self, library, k: int = 10) -> list[int]:
        cands = self.ease.ranking(library, self.n)
        score = {a: 1 - i / len(cands) + self.beta * math.log1p(self.recent.get(a, 0)) / self.top
                 for i, a in enumerate(cands)}
        return top_up(sorted(cands, key=lambda a: -score[a]), self.fallback, library, k)


def by_recency(entries: list[dict], liked_on: dict, half_life: float, cutoff: str = CUTOFF) -> list[dict]:
    """The same users with each visible game weighted by how long before the cutoff they liked it, half the weight
    per half-life: what a user liked lately says more about what they like next."""
    end = np.datetime64(cutoff)
    return [{**e, "visible": {a: 0.5 ** ((end - liked_on[(e["user"], a)]).astype(int) / half_life) for a in e["visible"]}}
            for e in entries]


class SavedSasrec:
    """SASRec's saved top 100 per eval user (eval/sasrec.py), looked up by the user's games, so the page's
    recs.sasrec_np.SasrecNudge can score the time split without torch here."""

    def __init__(self, top: dict[frozenset, list[int]]):
        self.top = top

    def ranking(self, ordered: list[int], n: int) -> list[int]:
        return self.top.get(frozenset(ordered), [])[:n]


def build(name: str, systems: Systems, users, apps, hours, dates, keep) -> object:
    """The time-aware systems; anything else comes from eval.run_eval. keep: training rows (before the cutoff,
    no eval user)."""
    cutoff = np.datetime64(CUTOFF)
    if name.startswith("pop-recent:"):  # "pop-recent:<days>"
        lately = keep & (dates >= cutoff - np.timedelta64(int(name.split(":")[1]), "D"))
        p = Popularity(build_interactions(users[lately], apps[lately], hours[lately], exclude_users=set()))
        p.name = name
        return p
    if name.startswith("ease-decay:"):  # "ease-decay:<half-life in days>": a like loses half its weight per half-life
        age = (cutoff - dates[keep]).astype(int)
        weight = 0.5 ** (age / float(name.split(":")[1]))
        e = EASE(build_interactions(users[keep], apps[keep], weight, exclude_users=set()), binary=False,
                 fallback=systems.popularity.ranking)
        e.name = name
        return e
    if name.startswith("ease-userdecay:"):  # weights come in as the library's values (see by_recency in main)
        e = EASE.__new__(EASE)
        e.__dict__ = dict(systems.get("ease").__dict__, name=name, weigh=lambda library: library)
        return e
    if name == "markov" or name.startswith("ease+markov:"):  # the library's values are recency weights (main)
        if "markov" not in systems.cache:
            train = systems.train
            systems.cache["markov"] = Markov(transitions(users[keep], apps[keep], dates[keep], train.item_index),
                                             train.item_ids, train.item_index, fallback=systems.popularity.ranking)
        if name == "markov":
            return systems.cache["markov"]
        return MarkovNudge(systems.get("ease"), systems.cache["markov"], float(name.split(":")[1]),
                           fallback=systems.popularity.ranking)
    if name.startswith("ease+sasrec:"):  # "ease+sasrec:<gamma>": needs a sasrec run on the same set first
        part = "test" if systems.cache.get("set") == "test" else "val"
        path = sorted(RESULTS.glob(f"*-sasrec-{part}.json"))[-1]
        shown = {e["user"]: frozenset(e["visible"]) for e in systems.cache["entries"]}
        top = {shown[u["user"]]: u["top100"] for u in json.loads(path.read_text(encoding="utf-8"))["rows"][0]["per_user"]}
        print(f"  SASRec candidates from {path.name}", flush=True)
        nudge = SasrecNudge(systems.get("ease"), SavedSasrec(top), float(name.split(":")[1]),
                            fallback=systems.popularity.ranking)
        nudge.name = name
        return nudge
    if name.startswith("ease+recent:"):  # "ease+recent:<beta>"
        lately = keep & (dates >= cutoff - np.timedelta64(RECENT_DAYS, "D"))
        ids, counts = np.unique(apps[lately], return_counts=True)
        return RecentNudge(systems.get("ease"), dict(zip(ids.tolist(), counts.tolist())), float(name.split(":")[1]),
                           fallback=systems.popularity.ranking)
    return systems.get(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--set", choices=["val", "test"], default="val")
    ap.add_argument("--systems", default="popularity,content,als,ease",
                    help="also pop-recent:<days>, ease-decay:<half-life days>, ease+recent:<beta>, "
                         "ease-userdecay:<half-life days>, markov, ease+markov:<gamma>")
    args = ap.parse_args()

    users, apps, hours, dates = positive_dates()
    if args.make:
        try:
            save_split(make_time_split(users, apps, hours, dates), TIME_SPLIT_PATH)
        except FileExistsError as e:
            raise SystemExit(str(e))
        print(f"wrote {TIME_SPLIT_PATH}")
        return

    split = load_split(TIME_SPLIT_PATH)
    eval_users = {e["user"] for part in split.values() for e in part}
    before = dates < np.datetime64(CUTOFF)
    keep = before & ~np.isin(users, list(eval_users))
    train = build_interactions(users[keep], apps[keep], hours[keep], exclude_users=set())
    systems = Systems(train, load_catalog())
    pop_rank = {app: i + 1 for i, app in enumerate(systems.popularity.ranking)}
    entries = split[args.set]
    systems.cache.update({"set": args.set, "entries": entries})  # ease+sasrec finds its own run by these
    hidden = [a for e in entries for a in e["hidden"]]
    unknown = sum(a not in train.item_index for a in hidden) / len(hidden)
    print(f"trained on positives before {CUTOFF}: {train.matrix.shape[0]} users x {train.matrix.shape[1]} games; "
          f"{unknown:.0%} of the hidden games are unknown to training (released or first liked later)", flush=True)
    rows = []
    mine = np.isin(users, list(eval_users))
    liked_on = dict(zip(zip(users[mine].tolist(), apps[mine].tolist()), dates[mine]))
    for name in args.systems.split(","):
        shown = (by_recency(entries, liked_on, float(name.split(":")[1])) if name.startswith("ease-userdecay:")
                 else by_recency(entries, liked_on, HALF_LIFE) if "markov" in name else entries)
        rows.append(evaluate(build(name, systems, users, apps, hours, dates, keep), shown, pop_rank))
        print(f"  {name}: recall@10 {rows[-1]['recall@10']:.3f}", flush=True)
    print(f"time split, {args.set}, {len(entries)} users")
    print(table(rows))
    for r in rows:  # the same, counting only hidden games training has seen: what no model could find is left out
        known = [recall_at_k(u["recs"], [a for a in u["hidden"] if a in train.item_index]) for u in r["per_user"]
                 if any(a in train.item_index for a in u["hidden"])]
        print(f"{r['system']}: recall@10 {sum(known) / len(known):.3f} on games training knows ({len(known)} users)")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-time-{args.set}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}), encoding="utf-8")
    print(f"per-user results: {out}")


if __name__ == "__main__":
    main()
