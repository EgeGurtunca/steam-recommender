"""LightGCN (He et al., 2020) on the time split: game and player embeddings smoothed over the like graph,
the graph-network version of the co-play graph that stores use for recommendations.

Trained with BPR (a liked game should score above a random one) on the same training likes as eval/sasrec.py
(data/build/seq.npz, written by `python -m eval.sasrec --export`). Eval users were never in the graph: a new
user is folded in as the mean of their games' final embeddings, so no retraining per user, as on the page.

Run (needs torch): <python with torch> -m eval.lightgcn --dim 64 --layers 3 --epochs 40 --batch 131072 --lr 0.01
Scores val and test after one training run (settings fixed before either is seen) and writes
eval/results/<date>-lightgcn-<set>.json for eval.intervals.
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # not recs.data: the torch side has no duckdb
SEQ = ROOT / "data" / "build" / "seq.npz"
RESULTS = ROOT / "eval" / "results"


def main() -> None:
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", default="val,test", help="scored after one training run, settings fixed beforehand")
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--reg", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    d = dict(np.load(SEQ))  # read whole: a later re-export must not change what this run sees
    item_ids, items, starts = d["item_ids"], d["train_items"] - 1, d["train_starts"]  # items 0-based here
    n_items, n_users = len(item_ids), len(starts)
    users = np.repeat(np.arange(n_users), np.diff(np.r_[starts, len(items)]))
    # the symmetric normalised user-game graph, D^-1/2 A D^-1/2, users first then games
    deg_u = np.bincount(users, minlength=n_users).astype(np.float32)
    deg_i = np.bincount(items, minlength=n_items).astype(np.float32)
    w = 1 / np.sqrt(deg_u[users] * np.maximum(deg_i[items], 1))
    rows = np.r_[users, n_users + items]
    cols = np.r_[n_users + items, users]
    graph = torch.sparse_coo_tensor(torch.from_numpy(np.stack([rows, cols])), torch.from_numpy(np.r_[w, w]),
                                    (n_users + n_items, n_users + n_items)).coalesce().to(dev)
    emb = torch.nn.Parameter(torch.randn(n_users + n_items, args.dim, device=dev) * 0.1)
    opt = torch.optim.Adam([emb], lr=args.lr)

    def propagate(keep_layers=False):
        x, out, layers = emb, emb, [emb]
        for _ in range(args.layers):
            x = torch.sparse.mm(graph, x)
            out = out + x
            layers.append(x)
        return layers if keep_layers else out / (args.layers + 1)

    pairs_u, pairs_i = torch.from_numpy(users).to(dev), torch.from_numpy(items).to(dev)
    steps = len(items) // args.batch
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(items), device=dev)
        total = 0.0
        for s in range(steps):
            b = order[s * args.batch:(s + 1) * args.batch]
            u, pos = pairs_u[b], pairs_i[b]
            neg = torch.randint(0, n_items, (len(b),), device=dev)  # ponytail: uniform negatives, may hit a liked game
            final = propagate()
            eu, ep, en = final[u], final[n_users + pos], final[n_users + neg]
            loss = -torch.nn.functional.logsigmoid((eu * ep).sum(1) - (eu * en).sum(1)).mean()
            loss = loss + args.reg * (emb[u].pow(2).sum() + emb[n_users + pos].pow(2).sum()
                                      + emb[n_users + neg].pow(2).sum()) / len(b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if s % 20 == 0:  # one overall count, which the progress pane reads for the time left
                print(f"PROGRESS {(epoch - 1) * steps + s}/{args.epochs * steps} (epoch {epoch})", flush=True)
        print(f"epoch {epoch} of {args.epochs}: loss {total / steps:.4f}", flush=True)

    with torch.no_grad():
        layers = [x[n_users:].cpu().numpy() for x in propagate(keep_layers=True)]
    games = sum(layers) / (args.layers + 1)  # each game's final embedding, as in training
    index = {int(a): i for i, a in enumerate(item_ids)}
    popular = np.argsort(-deg_i)
    inv_sqrt = 1 / np.sqrt(np.maximum(deg_i, 1))
    name = f"lightgcn:{args.dim}x{args.layers}"
    for part in args.sets.split(","):
        visible = json.loads(str(d[f"{part}_visible"]))
        hidden = json.loads(str(d[f"{part}_hidden"]))
        per_user = []
        for u, seen, hid in zip(d[f"{part}_users"], visible, hidden):
            known = [index[a] for a in seen if a in index]
            if known:  # the new user's layers by the same rule as a trained one's, with no layer-0 vector of its own
                w = inv_sqrt[known] / np.sqrt(len(known))
                user = sum(w @ layers[k][known] for k in range(args.layers)) / (args.layers + 1)
                s = user @ games.T
            else:
                s = -np.argsort(popular).astype(np.float32)
            owned = set(seen)
            ranked = [int(item_ids[j]) for j in np.argsort(-s)[:100 + len(owned)] if int(item_ids[j]) not in owned]
            per_user.append({"user": int(u), "recs": ranked[:10], "hidden": hid, "top100": ranked[:100]})
        recall = np.mean([len(set(p["recs"]) & set(p["hidden"])) / min(10, len(p["hidden"])) for p in per_user])
        print(f"{name}, time split {part}: recall@10 {recall:.3f} ({len(per_user)} users)")
        RESULTS.mkdir(exist_ok=True)
        out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-lightgcn-{part}.json"
        out.write_text(json.dumps({"args": vars(args), "rows": [{"system": name, "per_user": per_user}]}),
                       encoding="utf-8")
        print(f"per-user results: {out}")


if __name__ == "__main__":
    main()
