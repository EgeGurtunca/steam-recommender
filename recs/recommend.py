"""Ten recommendations with reasons, for one library, ranked as the page ranks (recs.web.page_ranker).

Run: python -m recs.recommend --library lib.json   ({"app_id": hours, ...} in the order played, oldest first)
     python -m recs.recommend --user <id>          (a frozen eval user's visible games; the page's model has seen
                                                    every user, so this shows the output, it is not an eval)
"""
import argparse
import json
from pathlib import Path

from recs import llm
from recs.data import BUILD_DIR, load_catalog
from recs.ease import EASE
from recs.explain import Explainer
from recs.split import load_split
from recs.web import page_ranker


class InOrder:
    """The page's ranker reads a library's values as recency; a JSON library gives its order instead."""

    def __init__(self, ranker):
        self.ranker, self.name = ranker, "page ranker"

    def recommend(self, library, k: int = 10) -> list[int]:
        return self.ranker.recommend({a: i for i, a in enumerate(library, 1)}, k)


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
    explainer = Explainer(InOrder(page_ranker(EASE.load_or_build(BUILD_DIR / "ease"))), catalog)
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
