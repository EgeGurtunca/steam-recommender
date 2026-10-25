"""Score recommenders on the frozen split: for each eval user, can the system find the games hidden from it?

Run: python -m eval.run_eval --set val --systems popularity,content,als,hybrid --users 1000
     python -m eval.run_eval --set test ... (only for the final numbers; tune on val)
Every user's recommendations and hidden games go to eval/results/<date>-<set>.json (not committed).
"""
import argparse
import json
import time
from contextlib import nullcontext
from datetime import datetime

from eval import metrics
from recs.als import ALS
from recs.content import Content
from recs.ease import EASE
from recs.hybrid import Hybrid
from recs import llm
from recs.rerank import LLMRerank
from recs.freeform import FreeFormLLM
from recs.data import ROOT, Catalog, Interactions, build_interactions, load_all_titles, load_catalog, load_positives, load_store
from recs.popularity import Popularity
from recs.split import load_split

RESULTS = ROOT / "eval" / "results"
K = 10
NEW_FROM = "2022-07-01"


class Systems:
    """Builds each system once; later systems reuse earlier ones (hybrid needs ALS and content)."""

    def __init__(self, train: Interactions, catalog: Catalog):
        self.train, self.catalog, self.cache = train, catalog, {}
        self.popularity = Popularity(train)

    def get(self, name: str):
        if name not in self.cache:
            self.cache[name] = self.build(name)
        return self.cache[name]

    def build(self, name: str):
        if name == "popularity":
            return self.popularity
        if name.startswith("content+pop"):  # "content+pop:<beta>": 2026 store tags, times log1p(positive reviews)**beta
            beta = float(name.split(":")[1]) if ":" in name else 1.0
            store = load_store()
            tags = {a: (store[a]["tags"] or t) if a in store else t for a, t in self.catalog.tags.items()}
            cat = Catalog(self.catalog.titles, tags, self.catalog.released)
            c = Content(cat, fallback=self.popularity.ranking, prior={a: v["positive"] for a, v in store.items()}, beta=beta)
            c.name = name
            return c
        if name == "content":
            return Content(self.catalog, fallback=self.popularity.ranking)
        if name == "als":
            return ALS(self.train, fallback=self.popularity.ranking)
        if name.startswith("ease-w:"):  # "ease-w:<mode>": rows weighted by recs.profile.engagement
            from recs.profile import engagement
            store, mode = load_store(), name.split(":")[1]
            e = EASE.__new__(EASE)
            e.__dict__ = dict(self.get("ease").__dict__, name=name, weigh=lambda lib: engagement(lib, store, mode))
            return e
        if name.startswith("ease"):  # "ease" or "ease:<lam>" for tuning on val
            lam = float(name.split(":")[1]) if ":" in name else 500.0
            return EASE(self.train, lam=lam, fallback=self.popularity.ranking)
        if name == "hybrid":
            return Hybrid(self.get("als"), self.get("content"), fallback=self.popularity.ranking)
        if name == "rerank":
            return LLMRerank(self.get("hybrid"), self.catalog)
        if name == "rerank-als20":
            als20 = Hybrid(self.get("als"), None, n_candidates=20, fallback=self.popularity.ranking)
            return LLMRerank(als20, self.catalog, name="als top20+gemma4")
        if name == "freeform":
            return FreeFormLLM(self.catalog, all_titles=load_all_titles())
        raise SystemExit(f"unknown system: {name}")


def evaluate(system, entries: list[dict], pop_rank: dict[int, int]) -> dict:
    per_user, start = [], time.perf_counter()
    for n, e in enumerate(entries, 1):
        per_user.append({"user": e["user"], "recs": system.recommend(e["visible"], K), "hidden": e["hidden"]})
        if n % 20 == 0 and getattr(system, "failures", None) is not None:  # LLM systems are slow: show progress
            print(f"PROGRESS {system.name} {n}/{len(entries)} "
                  f"recall@10 so far {sum(metrics.recall_at_k(u['recs'], u['hidden'], K) for u in per_user) / n:.3f}",
                  flush=True)
    seconds = (time.perf_counter() - start) / max(len(entries), 1)
    recs = [u["recs"] for u in per_user]
    return {
        "system": system.name,
        "recall@10": sum(metrics.recall_at_k(u["recs"], u["hidden"], K) for u in per_user) / len(per_user),
        "ndcg@10": sum(metrics.ndcg_at_k(u["recs"], u["hidden"], K) for u in per_user) / len(per_user),
        "hit@10": sum(bool(set(u["recs"][:K]) & set(u["hidden"])) for u in per_user) / len(per_user),
        "coverage": metrics.coverage(recs),
        "mean pop rank": metrics.mean_pop_rank(recs, pop_rank),
        "s/user": seconds,
        "per_user": per_user,
    }


def table(rows: list[dict]) -> str:
    cols = ["system", "recall@10", "ndcg@10", "hit@10", "coverage", "mean pop rank", "s/user"]
    fmt = {"recall@10": "{:.3f}", "ndcg@10": "{:.3f}", "hit@10": "{:.3f}", "coverage": "{}", "mean pop rank": "{:.0f}", "s/user": "{:.3f}"}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join(fmt.get(c, "{}").format(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], default="val")
    ap.add_argument("--systems", default="popularity")
    ap.add_argument("--users", type=int, default=1000)
    ap.add_argument("--cold-start", action="store_true",
                    help=f"treat games released from {NEW_FROM} as new: out of training, scored alone")
    args = ap.parse_args()

    split = load_split()
    eval_users = {e["user"] for part in split.values() for e in part}
    catalog = load_catalog()
    new_apps = {a for a, d in catalog.released.items() if d >= NEW_FROM} if args.cold_start else set()
    train = build_interactions(*load_positives(), exclude_users=eval_users, exclude_apps=frozenset(new_apps))
    systems = Systems(train, catalog)
    pop_rank = {app: i + 1 for i, app in enumerate(systems.popularity.ranking)}
    entries = split[args.set][: args.users]
    if args.cold_start:
        entries = metrics.cold_start_entries(entries, new_apps)
        print(f"cold start: {len(new_apps)} games released from {NEW_FROM} left out of training; "
              f"{len(entries)} users have one hidden")

    names = args.systems.split(",")
    with llm.ollama() if {"rerank", "rerank-als20", "freeform"} & set(names) else nullcontext():
        rows = [evaluate(systems.get(name), entries, pop_rank) for name in names]
    for name in names:
        if hasattr(systems.get(name), "failures"):
            print(f"{name}: {systems.get(name).failures} failed LLM calls (fell back)")
        if hasattr(systems.get(name), "totals"):
            t = systems.get(name).totals
            print(f"{name}: {t['titles']} titles named, {t['on_steam']} exist in the Steam list, "
                  f"{t['matched']} in our catalog, {t['owned']} already owned")
    print(f"{args.set}, {len(entries)} users, k={K}, train: {train.matrix.shape[0]} users x {train.matrix.shape[1]} games")
    print(table(rows))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-{args.set}.json"
    out.write_text(json.dumps({"args": vars(args), "rows": rows}), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
