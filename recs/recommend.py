"""Ten recommendations with reasons, for one library.

Run: python -m recs.recommend --user <id>          (a frozen eval user: only their visible games are used)
     python -m recs.recommend --library lib.json   ({"app_id": hours, ...})
"""
import argparse
import json
from pathlib import Path

from recs import llm
from recs.als import ALS
from recs.data import build_interactions, load_catalog, load_positives
from recs.explain import Explainer
from recs.popularity import Popularity
from recs.split import load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", type=int)
    ap.add_argument("--library", type=Path)
    ap.add_argument("-k", type=int, default=10)
    args = ap.parse_args()
    split = load_split()
    if args.library:
        library = {int(a): float(h) for a, h in json.loads(args.library.read_text(encoding="utf-8")).items()}
    else:
        library = next(e["visible"] for p in split.values() for e in p if e["user"] == args.user)
    catalog = load_catalog()
    # ponytail: retrains ALS (~95 s) on every call; save the factors if this becomes a page or a bot
    train = build_interactions(*load_positives(), exclude_users={e["user"] for p in split.values() for e in p})
    explainer = Explainer(ALS(train, fallback=Popularity(train).ranking), catalog)
    print("Played most:")
    for a, h in sorted(library.items(), key=lambda x: -x[1])[:5]:
        print(f"  {catalog.titles.get(a, a)}: {h:.0f} h")
    with llm.ollama():
        picks = explainer.explain(library, args.k)
    print("\nRecommended:")
    for i, (a, reason) in enumerate(picks, 1):
        print(f"{i:2}. {catalog.titles.get(a, a)}\n    {reason}")


if __name__ == "__main__":
    main()
