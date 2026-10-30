"""The frozen eval users: written once to eval/split.json and committed, so the test set cannot drift.

Run: python -m recs.split   (refuses to overwrite an existing split unless --force)
"""
import json
import sys
from pathlib import Path

import numpy as np

from recs.data import ROOT, load_positives

SPLIT_PATH = ROOT / "eval" / "split.json"


def make_split(users: np.ndarray, apps: np.ndarray, hours: np.ndarray, n_val: int = 1000, n_test: int = 1000,
               min_pos: int = 10, hide: float = 0.2, min_hidden: int = 2, seed: int = 42) -> dict:
    """users must be sorted (as in positives.parquet). Each picked user: 20% of positives hidden, at least 2."""
    rng = np.random.default_rng(seed)
    ids, starts, counts = np.unique(users, return_index=True, return_counts=True)
    eligible = np.flatnonzero(counts >= min_pos)
    picked = rng.choice(eligible, size=n_val + n_test, replace=False)
    out = {"val": [], "test": []}
    for n, i in enumerate(picked):
        rows = slice(starts[i], starts[i] + counts[i])
        owned = dict(zip(apps[rows].tolist(), hours[rows].tolist()))
        hidden = sorted(rng.choice(list(owned), size=max(min_hidden, round(hide * len(owned))), replace=False).tolist())
        visible = {a: h for a, h in owned.items() if a not in set(hidden)}
        out["val" if n < n_val else "test"].append({"user": int(ids[i]), "visible": visible, "hidden": hidden})
    return out


def save_split(split: dict, path: Path = SPLIT_PATH, force: bool = False) -> None:
    if path.exists() and not force:
        raise FileExistsError(f"{path} exists: the test set is frozen (use --force only before any result is reported)")
    path.write_text(json.dumps(split), encoding="utf-8")


def load_split(path: Path = SPLIT_PATH) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {part: [{"user": e["user"], "visible": {int(a): h for a, h in e["visible"].items()}, "hidden": e["hidden"]}
                   for e in entries] for part, entries in raw.items()}


if __name__ == "__main__":
    try:
        save_split(make_split(*load_positives()), force="--force" in sys.argv)
    except FileExistsError as e:
        raise SystemExit(str(e)) from None
    print(f"wrote {SPLIT_PATH}")
