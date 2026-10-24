"""How grounded are gemma4's reasons? Share of sentences that name a game the player actually played.

Run: python -m eval.explain_eval --set test --users 100
"""
import argparse
import json
import time
from datetime import datetime

from eval.run_eval import RESULTS
from recs import llm
from recs.als import ALS
from recs.data import build_interactions, load_catalog, load_positives
from recs.explain import Explainer, grounded
from recs.popularity import Popularity
from recs.split import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["val", "test"], default="val")
    ap.add_argument("--users", type=int, default=100)
    args = ap.parse_args()
    split, catalog = load_split(), load_catalog()
    train = build_interactions(*load_positives(), exclude_users={e["user"] for p in split.values() for e in p})
    explainer = Explainer(ALS(train, fallback=Popularity(train).ranking), catalog)
    rows, start = [], time.perf_counter()
    with llm.ollama():
        for n, e in enumerate(split[args.set][: args.users], 1):
            picks = explainer.explain(e["visible"], 10)
            rows.append({"user": e["user"], "picks": [{"app": a, "title": catalog.titles.get(a), "reason": r,
                                                       "grounded": grounded(r, e["visible"], catalog)} for a, r in picks]})
            if n % 20 == 0:
                done = [p for r in rows for p in r["picks"]]
                print(f"PROGRESS {n}/{args.users} grounded so far {sum(p['grounded'] for p in done) / len(done):.2f}",
                      flush=True)
    picks = [p for r in rows for p in r["picks"]]
    print(f"{args.set}, {len(rows)} users, {len(picks)} reasons: grounded {sum(p['grounded'] for p in picks) / len(picks):.2f}, "
          f"empty {sum(not p['reason'] for p in picks) / len(picks):.2f}, failed calls {explainer.failures}, "
          f"{(time.perf_counter() - start) / len(rows):.1f} s/user")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-explain-{args.set}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
