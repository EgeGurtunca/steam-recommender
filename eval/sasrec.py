"""SASRec (Kang and McAuley, 2018) on the time split: a small transformer reads a library in date order and
predicts the next like, against the transition counts in recs/sequence.py.

Two steps, so the main environment needs no torch:
  python -m eval.sasrec --export              (this project's env: writes data/build/seq.npz from the time split)
  python -m eval.sasrec --export --page       (every positive, for the page) then <torch python> -m eval.sasrec --page
  <python with torch> -m eval.sasrec --set val (trains on the training users' sequences, scores the eval users,
                                                and writes eval/results/<date>-sasrec-<set>.json for eval.intervals)
"""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]  # not recs.data: the torch side has no duckdb
SEQ = ROOT / "data" / "build" / "seq.npz"
SEQ_ALL = ROOT / "data" / "build" / "seq_all.npz"  # every positive, for the page's model
PAGE_MODEL = ROOT / "data" / "build" / "sasrec_page.npz"
RESULTS = ROOT / "eval" / "results"
MAX_LEN = 50


def export(everything: bool = False) -> None:
    """The time split's training sequences and eval users, or with everything=True every positive (the page)."""
    from eval.time_split import CUTOFF, TIME_SPLIT_PATH
    from recs.data import load_positive_dates
    from recs.split import load_split

    users, apps, _, dates = load_positive_dates()
    split = {} if everything else load_split(TIME_SPLIT_PATH)
    eval_users = {e["user"] for part in split.values() for e in part}
    keep = np.ones(len(users), bool) if everything else (dates < np.datetime64(CUTOFF)) & ~np.isin(users, list(eval_users))
    item_ids = np.unique(apps[keep])
    index = {int(a): i + 1 for i, a in enumerate(item_ids)}  # 0 is padding
    order = np.lexsort((dates[keep], users[keep]))
    u, a = users[keep][order], apps[keep][order]
    starts = np.flatnonzero(np.r_[True, u[1:] != u[:-1]])
    mine = np.isin(users, list(eval_users))
    liked_on = dict(zip(zip(users[mine].tolist(), apps[mine].tolist()), dates[mine]))
    out = {"item_ids": item_ids, "train_items": np.searchsorted(item_ids, a) + 1, "train_starts": starts}
    for part in split:
        seqs, hidden = [], []
        for e in split[part]:
            seen = sorted(e["visible"], key=lambda x: liked_on[(e["user"], x)])
            seqs.append([index[x] for x in seen if x in index][-MAX_LEN:])
            hidden.append(e["hidden"])
        out[f"{part}_users"] = np.array([e["user"] for e in split[part]])
        out[f"{part}_seq"] = np.array([s + [0] * (MAX_LEN - len(s)) for s in seqs], dtype=np.int64)
        out[f"{part}_len"] = np.array([len(s) for s in seqs])
        out[f"{part}_visible"] = np.array(json.dumps([list(e["visible"]) for e in split[part]]))
        out[f"{part}_hidden"] = np.array(json.dumps(hidden))
    path = SEQ_ALL if everything else SEQ
    np.savez_compressed(path, **out)
    print(f"wrote {path}: {len(item_ids)} games, {len(starts)} training users")


def train_and_score(part: str | None, epochs: int, dim: int, seed: int, save=None, layers: int = 2) -> None:
    """part None: train on every positive (seq_all.npz) and only save the weights, for recs/sasrec_np.py."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = np.load(SEQ if part else SEQ_ALL)
    items, starts = d["train_items"], d["train_starts"]
    ends = np.r_[starts[1:], len(items)]
    usable = np.flatnonzero(ends - starts >= 2)
    n = len(d["item_ids"])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    class SASRec(nn.Module):
        def __init__(self):
            super().__init__()
            self.item = nn.Embedding(n + 1, dim, padding_idx=0)
            self.pos = nn.Embedding(MAX_LEN, dim)
            layer = nn.TransformerEncoderLayer(dim, 2, dim * 4, 0.2, batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers)
            self.norm = nn.LayerNorm(dim)
            self.causal = nn.Transformer.generate_square_subsequent_mask(MAX_LEN)
            for e in (self.item, self.pos):  # the default N(0, 1) times sqrt(dim) makes the first logits explode
                nn.init.normal_(e.weight, std=0.02)

        def forward(self, seq):  # right-padded: with the causal mask a real step never sees padding
            x = self.item(seq) + self.pos.weight[None, : seq.shape[1]]
            mask = self.causal[: seq.shape[1], : seq.shape[1]].to(seq.device)
            return self.norm(self.enc(x, mask=mask, is_causal=True))

        def logits(self, h):
            return h @ self.item.weight[1:].T  # every game, padding left out

    model = SASRec().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    batch = 512
    for epoch in range(1, epochs + 1):
        rng.shuffle(usable)
        model.train()
        total, steps = 0.0, len(usable) // batch
        for b in range(steps):
            rows = usable[b * batch:(b + 1) * batch]
            seq = np.zeros((len(rows), MAX_LEN + 1), dtype=np.int64)
            for r, i in enumerate(rows):
                s = items[starts[i]:ends[i]]
                if len(s) > MAX_LEN + 1:  # a random window of a long library
                    o = rng.integers(0, len(s) - MAX_LEN)
                    s = s[o:o + MAX_LEN + 1]
                seq[r, :len(s)] = s
            seq = torch.from_numpy(seq).to(dev)
            inp, target = seq[:, :-1], seq[:, 1:] - 1  # target -1: padding, ignored
            with torch.autocast(dev, dtype=torch.bfloat16):
                loss = nn.functional.cross_entropy(model.logits(model(inp)).flatten(0, 1).float(), target.flatten(),
                                                   ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            if b % 100 == 0:  # one overall count, which the progress pane reads for the time left
                print(f"PROGRESS {(epoch - 1) * steps + b}/{epochs * steps} (epoch {epoch}, loss {total / (b + 1):.3f})",
                      flush=True)

    model.eval()
    item_ids = d["item_ids"]
    if save:  # weights by their torch names, plus a few users' scores so the numpy port can be checked
        weights = {k: v.float().cpu().numpy() for k, v in model.state_dict().items()}
        check = d[f"{part}_seq"][:8] if part else np.array([items[starts[i]:ends[i]][-MAX_LEN:].tolist() + [0] * (
            MAX_LEN - min(MAX_LEN, ends[i] - starts[i])) for i in usable[:8]])
        with torch.no_grad():
            h = model(torch.from_numpy(check).to(dev))
            lens = (check > 0).sum(axis=1)
            last = model.logits(h[torch.arange(len(check)), torch.from_numpy(lens - 1).to(dev)]).float().cpu().numpy()
        np.savez_compressed(save, item_ids=item_ids, dim=dim, check_seq=check, check_scores=last, **weights)
        print(f"saved weights: {save}")
    if not part:
        return
    seqs, lens = d[f"{part}_seq"], d[f"{part}_len"]
    visible, hidden = json.loads(str(d[f"{part}_visible"])), json.loads(str(d[f"{part}_hidden"]))
    popular = np.argsort(-np.bincount(items, minlength=n + 1)[1:])
    per_user = []
    with torch.no_grad():
        for start in range(0, len(seqs), 256):
            seq = torch.from_numpy(seqs[start:start + 256]).to(dev)
            h = model(seq)
            last = torch.from_numpy(np.maximum(lens[start:start + 256] - 1, 0)).to(dev)
            scores = model.logits(h[torch.arange(len(seq)), last]).float().cpu().numpy()
            for r in range(len(seq)):
                i = start + r
                s = scores[r] if lens[i] else -np.arange(n, dtype=np.float32)[np.argsort(popular)]
                owned = set(visible[i])
                ranked = [int(item_ids[j]) for j in np.argsort(-s)[:100 + len(owned)] if int(item_ids[j]) not in owned]
                per_user.append({"user": int(d[f"{part}_users"][i]), "recs": ranked[:10], "hidden": hidden[i],
                                 "top100": ranked[:100]})  # for eval.time_split's ease+sasrec
    recall = np.mean([len(set(u["recs"]) & set(u["hidden"])) / min(10, len(u["hidden"])) for u in per_user])
    print(f"sasrec, time split {part}: recall@10 {recall:.3f} ({len(per_user)} users)")
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{datetime.now():%Y-%m-%d-%H%M}-sasrec-{part}.json"
    out.write_text(json.dumps({"args": {"set": part, "epochs": epochs, "dim": dim},
                               "rows": [{"system": f"sasrec:{dim}" + (f"x{layers}" if layers != 2 else ""),
                                         "per_user": per_user}]}), encoding="utf-8")
    print(f"per-user results: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", action="store_true")
    ap.add_argument("--set", choices=["val", "test"], default="val")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--page", action="store_true", help="with --export: every positive; else train the page's model")
    ap.add_argument("--save", help="also save the weights here (numpy, for recs/sasrec_np.py)")
    args = ap.parse_args()
    if args.export:
        export(everything=args.page)
    elif args.page:
        train_and_score(None, args.epochs, args.dim, args.seed, save=args.save or PAGE_MODEL, layers=args.layers)
    else:
        train_and_score(args.set, args.epochs, args.dim, args.seed, save=args.save, layers=args.layers)


if __name__ == "__main__":
    main()
