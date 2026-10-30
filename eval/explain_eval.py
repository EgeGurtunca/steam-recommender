"""How grounded are gemma4's reasons? Share of sentences that name a game the player actually played.

Run: python -m eval.explain_eval --set test --users 100
     python -m eval.explain_eval --set test --users 100 --ranker page   (the page's picks, EASE + SASRec; also how
                                                                        often a reason names the card's "because")
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
    ap.add_argument("--ranker", choices=["als", "page"], default="als")
    args = ap.parse_args()
    split, catalog = load_split(), load_catalog()
    if args.ranker == "page":  # its model has seen every user: this measures the reasons, not the picks
        from recs.data import BUILD_DIR, load_store
        from recs.ease import EASE
        from recs.profile import is_game
        from recs.recommend import InOrder
        from recs.web import page_ranker
        page, store = page_ranker(EASE.load_or_build(BUILD_DIR / "ease")), load_store()
        ordered = InOrder(page)

        class Cards:  # what the page shows: programs (Wallpaper Engine and the like) dropped by their Steam genre
            name = "page"

            def recommend(self, library, k=10):
                return [a for a in ordered.recommend(library, k + 30) if is_game(a, store)][:k]

        explainer = Explainer(Cards(), catalog)
        because = page.ease.because
    else:
        train = build_interactions(*load_positives(), exclude_users={e["user"] for p in split.values() for e in p})
        explainer = Explainer(ALS(train, fallback=Popularity(train).ranking), catalog)
        because = None
    rows, start = [], time.perf_counter()
    with llm.ollama():
        for n, e in enumerate(split[args.set][: args.users], 1):
            picks = explainer.explain(e["visible"], 10)
            why = because(e["visible"], [a for a, _ in picks]) if because else {}
            rows.append({"user": e["user"], "picks": [
                {"app": a, "title": catalog.titles.get(a), "reason": r, "grounded": grounded(r, e["visible"], catalog),
                 "because": catalog.titles.get(why.get(a)),
                 "names_because": a in why and grounded(r, {why[a]: 1.0}, catalog)} for a, r in picks]})
            if n % 20 == 0:
                done = [p for r in rows for p in r["picks"]]
                print(f"PROGRESS {n}/{args.users} grounded so far {sum(p['grounded'] for p in done) / len(done):.2f}",
                      flush=True)
    picks = [p for r in rows for p in r["picks"]]
    grounded = sum(p["grounded"] for p in picks) / len(picks)
    print(f"{args.set}, {len(rows)} users, {len(picks)} reasons: grounded {grounded:.2f}, "
          f"empty {sum(not p['reason'] for p in picks) / len(picks):.2f}, failed calls {explainer.failures}, "
          f"{(time.perf_counter() - start) / len(rows):.1f} s/user")
    if because:
        print(f"reasons that name the card's exact 'because you played' game: "
              f"{sum(p['names_because'] for p in picks) / len(picks):.2f}")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-explain-{args.ranker}-{args.set}.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
