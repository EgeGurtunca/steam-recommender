"""95% intervals for recall@10 gaps between systems, from a saved eval/results file (the same users, paired).

Run: python -m eval.intervals eval/results/<file>.json ease als content+pop:1
     (the first system is the base; each other one is compared with it)
     python -m eval.intervals eval/results/<file>.json als ease --by-size    (per library size; random split only)
     python -m eval.intervals <time-val file>+<sasrec-val file> ease sasrec:64   (systems from two runs)
"""
import json
import sys

from eval import metrics
from recs.split import load_split

SIZES = [(10, 15), (15, 30), (30, None)]  # liked games, visible + hidden; every eval user has at least 10


def per_user_recall(row: dict) -> dict[int, float]:
    return {u["user"]: metrics.recall_at_k(u["recs"], u["hidden"]) for u in row["per_user"]}


def main() -> None:
    by_size = "--by-size" in sys.argv
    path, base, *others = [a for a in sys.argv[1:] if a != "--by-size"]
    rows = {r["system"]: r for f in path.split("+")  # several files: a+b.json (the same users); last name wins
            for r in json.load(open(f, encoding="utf-8"))["rows"]}
    b = per_user_recall(rows[base])
    groups = {"all": set(b)}
    if by_size:
        size = {e["user"]: len(e["visible"]) + len(e["hidden"]) for part in load_split().values() for e in part}
        if not set(b) <= set(size):
            raise SystemExit("--by-size reads library sizes from eval/split.json: use it on random split results")
        groups = {f"{lo} to {hi - 1}" if hi else f"{lo}+": {u for u in b if lo <= size[u] < (hi or 10**9)}
                  for lo, hi in SIZES}
    print(f"| system | liked games | users | {base} | system | gap | 95% interval |\n" + "|---" * 7 + "|")
    for name in others:
        o = per_user_recall(rows[name])
        for group, members in groups.items():
            users = sorted(members & o.keys())
            gap, lo, hi = metrics.paired_bootstrap([b[u] for u in users], [o[u] for u in users])
            print(f"| {name} | {group} | {len(users)} | {sum(b[u] for u in users) / len(users):.3f} | "
                  f"{sum(o[u] for u in users) / len(users):.3f} | {gap:+.3f} | {lo:+.3f} to {hi:+.3f} |")


if __name__ == "__main__":
    main()
