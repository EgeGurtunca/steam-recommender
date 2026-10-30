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

from eval.metrics import diversify, recall_at_k, tag_diversity
from eval.run_eval import RESULTS, Systems, evaluate, table
from recs.core import top_up
from recs.data import ROOT, build_interactions, load_catalog
from recs.data import load_positive_dates as positive_dates
from recs.ease import EASE
from recs.popularity import Popularity
from recs.sasrec_np import SasrecNudge
from recs.sequence import HALF_LIFE, Markov, MarkovNudge, transitions
from recs.split import load_split, save_split

CUTOFF = "2022-01-01"
TIME_SPLIT_PATH = ROOT / "eval" / "time_split.json"
# a third group, frozen after 13 systems had seen the test set: kept out of every training run, scored once for
# the headline claim (EASE + SASRec over EASE)
HOLDOUT_PATH = ROOT / "eval" / "time_holdout.json"
SASREC_TIME = ROOT / "data" / "build" / "sasrec_time.npz"  # eval/sasrec.py --save, trained on this split's training
RECENT_DAYS = 30  # "lately": the best window for recent popularity on val (30, 90, 180 tried)


def make_time_split(users: np.ndarray, apps: np.ndarray, hours: np.ndarray, dates: np.ndarray, cutoff: str = CUTOFF,
                    n_val: int = 1000, n_test: int = 1000, min_before: int = 8, min_after: int = 2,
                    seed: int = 42, exclude: frozenset = frozenset()) -> dict:
    """users sorted. Eligible: at least min_before positives before the cutoff and min_after on or after it, and
    not in `exclude` (users another group already holds)."""
    rng = np.random.default_rng(seed)
    after = dates >= np.datetime64(cutoff)
    ids, starts, counts = np.unique(users, return_index=True, return_counts=True)
    n_after = np.add.reduceat(after.astype(int), starts)
    eligible = np.flatnonzero((counts - n_after >= min_before) & (n_after >= min_after) & ~np.isin(ids, list(exclude)))
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


def by_recency(entries: list[dict], liked_on: dict, half_life: float, cutoff: str | None = None) -> list[dict]:
    """The same users with each visible game weighted by how long before the cutoff they liked it, half the weight
    per half-life: what a user liked lately says more about what they like next."""
    end = np.datetime64(cutoff or CUTOFF)
    return [{**e, "visible": {a: 0.5 ** ((end - liked_on[(e["user"], a)]).astype(int) / half_life)
                              for a in e["visible"]}} for e in entries]


class SasrecAlone:
    """SASRec on its own; the library's values are recency weights (by_recency), which give it the order."""
    name = "sasrec"

    def __init__(self, model, fallback=()):
        self.model, self.fallback = model, fallback

    def recommend(self, library, k: int = 10) -> list[int]:
        return top_up(self.model.ranking(sorted(library, key=library.get), k), self.fallback, library, k)


class Diverse:
    """Any system's top 60, picked again for tag variety (eval.metrics.diversify)."""

    def __init__(self, base, tags: dict, lam: float, fallback=()):
        self.base, self.tags, self.lam, self.fallback = base, tags, lam, fallback
        self.name = f"{base.name} div:{lam:g}"

    def recommend(self, library, k: int = 10) -> list[int]:
        ranked = self.base.ranking(library, 60) if hasattr(self.base, "ranking") else self.base.recommend(library, 60)
        return top_up(diversify(ranked, self.tags, self.lam, k), self.fallback, library, k)


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
    if name == "sasrec" or name.startswith("ease+sasrec-np:"):  # numpy SASRec trained on this split's training
        if "sasrec_np" not in systems.cache:
            from recs.sasrec_np import SASRec
            systems.cache["sasrec_np"] = SASRec(SASREC_TIME)
        model = systems.cache["sasrec_np"]
        if name == "sasrec":
            return SasrecAlone(model, fallback=systems.popularity.ranking)
        nudge = SasrecNudge(systems.get("ease"), model, float(name.split(":")[1]), fallback=systems.popularity.ranking)
        nudge.name = name
        return nudge
    if name.startswith("ease+sasrec:"):  # "ease+sasrec:<gamma>": needs a sasrec run on the same set first
        part = "test" if systems.cache.get("set") == "test" else "val"
        tag = "" if CUTOFF == "2022-01-01" else CUTOFF[:4] + "-"
        path = sorted(RESULTS.glob(f"*-sasrec-{tag}{part}.json"))[-1]
        shown = {e["user"]: frozenset(e["visible"]) for e in systems.cache["entries"]}
        run = json.loads(path.read_text(encoding="utf-8"))["rows"][0]["per_user"]
        top = {shown[u["user"]]: u["top100"] for u in run}
        print(f"  SASRec candidates from {path.name}", flush=True)
        nudge = SasrecNudge(systems.get("ease"), SavedSasrec(top), float(name.split(":")[1]),
                            fallback=systems.popularity.ranking)
        nudge.name = name
        return nudge
    if name.startswith("ease-top:"):  # "ease-top:<k>": each game keeps its k largest weights by size, in float16
        base, k = systems.get("ease"), int(name.split(":")[1])
        b = base.b.copy()
        if k < b.shape[1]:
            size = np.abs(b)
            b[size < np.partition(size, -k, axis=1)[:, -k:-k + 1]] = 0.0
        e = EASE.__new__(EASE)
        e.__dict__ = dict(base.__dict__, name=name, b=b.astype(np.float16).astype(np.float32))
        return e
    if name.startswith("ease+recent:"):  # "ease+recent:<beta>"
        lately = keep & (dates >= cutoff - np.timedelta64(RECENT_DAYS, "D"))
        ids, counts = np.unique(apps[lately], return_counts=True)
        return RecentNudge(systems.get("ease"), dict(zip(ids.tolist(), counts.tolist())), float(name.split(":")[1]),
                           fallback=systems.popularity.ranking)
    return systems.get(name)


def use_cutoff(cutoff: str) -> None:
    """Another deployment date: its own frozen split and holdout files (eval/sasrec.py --cutoff reads them too)."""
    global CUTOFF, TIME_SPLIT_PATH, HOLDOUT_PATH
    CUTOFF = cutoff
    TIME_SPLIT_PATH = ROOT / "eval" / f"time_split_{cutoff[:4]}.json"
    HOLDOUT_PATH = ROOT / "eval" / f"time_holdout_{cutoff[:4]}.json"


def load_groups() -> dict:
    """val and test, plus the holdout group once it exists."""
    groups = load_split(TIME_SPLIT_PATH)
    return groups | load_split(HOLDOUT_PATH) if HOLDOUT_PATH.exists() else groups


def latest(entries: list[dict], liked_on: dict, n: int) -> list[dict]:
    """The same users shown only their n most recently liked games: a new account, as the page often gets."""
    return [{**e, "visible": dict(sorted(e["visible"].items(), key=lambda kv: liked_on[(e["user"], kv[0])])[-n:])}
            for e in entries]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--make", action="store_true")
    ap.add_argument("--make-holdout", action="store_true")
    ap.add_argument("--set", choices=["val", "test", "holdout"], default="val")
    ap.add_argument("--last", type=int, help="show each user only their latest N liked games")
    ap.add_argument("--diversify", type=float, help="pick each system's top 60 again for tag variety (lambda)")
    ap.add_argument("--systems", default="popularity,content,als,ease",
                    help="also pop-recent:<days>, ease-decay:<half-life days>, ease+recent:<beta>, "
                         "ease-userdecay:<half-life days>, markov, ease+markov:<gamma>, sasrec, "
                         "ease+sasrec-np:<gamma>, ease-top:<k>")
    ap.add_argument("--cutoff", help="another deployment date (e.g. 2021-01-01): its own frozen split, the year "
                                     "after it hidden, everything later dropped")
    args = ap.parse_args()

    users, apps, hours, dates = positive_dates()
    if args.cutoff:
        use_cutoff(args.cutoff)
        world = dates < np.datetime64(CUTOFF) + np.timedelta64(365, "D")
        users, apps, hours, dates = users[world], apps[world], hours[world], dates[world]
    if args.make:
        try:
            save_split(make_time_split(users, apps, hours, dates, cutoff=CUTOFF), TIME_SPLIT_PATH)
        except FileExistsError as e:
            raise SystemExit(str(e)) from None
        print(f"wrote {TIME_SPLIT_PATH}")
        return
    if args.make_holdout:
        taken = frozenset(e["user"] for part in load_split(TIME_SPLIT_PATH).values() for e in part)
        group = make_time_split(users, apps, hours, dates, cutoff=CUTOFF, n_val=1000, n_test=0, seed=7,
                                exclude=taken)["val"]
        try:
            save_split({"holdout": group}, HOLDOUT_PATH)
        except FileExistsError as e:
            raise SystemExit(str(e)) from None
        print(f"wrote {HOLDOUT_PATH}")
        return

    split = load_groups()
    eval_users = {e["user"] for part in split.values() for e in part}
    before = dates < np.datetime64(CUTOFF)
    keep = before & ~np.isin(users, list(eval_users))
    train = build_interactions(users[keep], apps[keep], hours[keep], exclude_users=set())
    systems = Systems(train, load_catalog())
    pop_rank = {app: i + 1 for i, app in enumerate(systems.popularity.ranking)}
    mine = np.isin(users, list(eval_users))
    liked_on = dict(zip(zip(users[mine].tolist(), apps[mine].tolist()), dates[mine]))
    entries = split[args.set] if not args.last else latest(split[args.set], liked_on, args.last)
    systems.cache.update({"set": args.set, "entries": entries})  # ease+sasrec finds its own run by these
    hidden = [a for e in entries for a in e["hidden"]]
    unknown = sum(a not in train.item_index for a in hidden) / len(hidden)
    print(f"trained on positives before {CUTOFF}: {train.matrix.shape[0]} users x {train.matrix.shape[1]} games; "
          f"{unknown:.0%} of the hidden games are unknown to training (released or first liked later)", flush=True)
    rows = []
    for name in args.systems.split(","):
        shown = (by_recency(entries, liked_on, float(name.split(":")[1])) if name.startswith("ease-userdecay:")
                 else by_recency(entries, liked_on, HALF_LIFE) if "markov" in name or "sasrec" in name else entries)
        system = build(name, systems, users, apps, hours, dates, keep)
        if args.diversify is not None:
            system = Diverse(system, systems.catalog.tags, args.diversify, fallback=systems.popularity.ranking)
        rows.append(evaluate(system, shown, pop_rank))
        print(f"  {name}: recall@10 {rows[-1]['recall@10']:.3f}", flush=True)
    print(f"time split, {args.set}, {len(entries)} users")
    print(table(rows))
    for r in rows:
        variety = tag_diversity([u["recs"] for u in r["per_user"]], systems.catalog.tags)
        print(f"{r['system']}: tag diversity {variety:.3f}")
    for r in rows:  # the same, counting only hidden games training has seen: what no model could find is left out
        known = [recall_at_k(u["recs"], [a for a in u["hidden"] if a in train.item_index]) for u in r["per_user"]
                 if any(a in train.item_index for a in u["hidden"])]
        print(f"{r['system']}: recall@10 {sum(known) / len(known):.3f} on games training knows ({len(known)} users)")
    RESULTS.mkdir(exist_ok=True)
    year, last = f"-{CUTOFF[:4]}" if args.cutoff else "", f"-last{args.last}" if args.last else ""
    out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-time{year}-{args.set}{last}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}), encoding="utf-8")
    print(f"per-user results: {out}")


if __name__ == "__main__":
    main()
