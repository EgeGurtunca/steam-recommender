"""ALS settings, chosen on the validation users only (never on test).

Run: python -m eval.tune_als
"""
import itertools
import time

from eval.run_eval import evaluate
from recs.als import ALS
from recs.data import build_interactions, load_positives
from recs.popularity import Popularity
from recs.split import load_split

GRID = {"factors": [64, 128], "alpha": [1.0, 4.0]}


def main() -> None:
    split = load_split()
    train = build_interactions(*load_positives(), exclude_users={e["user"] for p in split.values() for e in p})
    pop = Popularity(train)
    pop_rank = {app: i + 1 for i, app in enumerate(pop.ranking)}
    print("| factors | alpha | recall@10 | ndcg@10 | coverage | train s |\n|---|---|---|---|---|---|")
    for factors, alpha in itertools.product(GRID["factors"], GRID["alpha"]):
        start = time.perf_counter()
        als = ALS(train, factors=factors, alpha=alpha, fallback=pop.ranking)
        seconds = time.perf_counter() - start
        r = evaluate(als, split["val"], pop_rank)
        print(f"| {factors} | {alpha} | {r['recall@10']:.3f} | {r['ndcg@10']:.3f} | {r['coverage']} | {seconds:.0f} |",
              flush=True)


if __name__ == "__main__":
    main()
