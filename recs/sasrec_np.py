"""SASRec's forward pass in numpy, so the page can use the model eval/sasrec.py trains without torch.

The layers mirror torch's TransformerEncoderLayer(norm_first=True, activation=relu) with dropout off:
x += attention(norm1(x)); x += linear2(relu(linear1(norm2(x)))); then a final norm. Weights come from
eval/sasrec.py --save (torch's own names), and the file carries a few users' torch scores to check against.
"""
import numpy as np

from recs.core import top_up

HEADS = 2  # as trained in eval/sasrec.py
MAX_LEN = 50


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mean = x.mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(x.var(-1, keepdims=True) + eps) * w + b


class SASRec:
    def __init__(self, path):
        f = np.load(path)
        self.w = {k: f[k] for k in f.files}
        self.item_ids = f["item_ids"]
        self.index = {int(a): i + 1 for i, a in enumerate(self.item_ids)}  # 0 is padding
        self.layers = sorted({int(k.split(".")[2]) for k in self.w if k.startswith("enc.layers.")})

    def hidden(self, seq: np.ndarray) -> np.ndarray:
        """seq: (batch, length) item indices, right-padded with 0. Returns (batch, length, dim)."""
        w, (b, n) = self.w, seq.shape
        x = w["item.weight"][seq] + w["pos.weight"][None, :n]
        causal = np.triu(np.full((n, n), -np.inf, dtype=np.float32), 1)
        for i in self.layers:
            p = f"enc.layers.{i}."
            h = layer_norm(x, w[p + "norm1.weight"], w[p + "norm1.bias"])
            qkv = h @ w[p + "self_attn.in_proj_weight"].T + w[p + "self_attn.in_proj_bias"]
            q, k, v = (t.reshape(b, n, HEADS, -1).transpose(0, 2, 1, 3) for t in np.split(qkv, 3, axis=-1))
            att = q @ k.transpose(0, 1, 3, 2) / np.sqrt(q.shape[-1]) + causal
            att = np.exp(att - att.max(-1, keepdims=True))
            att /= att.sum(-1, keepdims=True)
            out = (att @ v).transpose(0, 2, 1, 3).reshape(b, n, -1)
            x = x + out @ w[p + "self_attn.out_proj.weight"].T + w[p + "self_attn.out_proj.bias"]
            h = layer_norm(x, w[p + "norm2.weight"], w[p + "norm2.bias"])
            h = np.maximum(h @ w[p + "linear1.weight"].T + w[p + "linear1.bias"], 0)
            x = x + h @ w[p + "linear2.weight"].T + w[p + "linear2.bias"]
        return layer_norm(x, w["norm.weight"], w["norm.bias"])

    def scores_for(self, seq: np.ndarray) -> np.ndarray:
        """Each row's next-like scores over item_ids, read at its last real step."""
        last = np.maximum((seq > 0).sum(axis=1) - 1, 0)
        h = self.hidden(seq)[np.arange(len(seq)), last]
        return h @ self.w["item.weight"][1:].T

    def ranking(self, ordered: list[int], n: int) -> list[int]:
        """ordered: a library's games oldest first. The n best next games it does not own."""
        idx = [self.index[a] for a in ordered if a in self.index][-MAX_LEN:]
        if not idx:
            return []
        s = self.scores_for(np.array([idx + [0] * (MAX_LEN - len(idx))]))[0]
        owned = set(ordered)
        top = (int(self.item_ids[j]) for j in np.argsort(-s)[: n + len(owned)])
        return [a for a in top if a not in owned][:n]


class SasrecNudge:
    """EASE's top 100, each moved up by its place in SASRec's top 100: the best pair on the time split's test.
    The library's values are recency weights (recs.web.recency), which give SASRec its order."""

    def __init__(self, ease, sasrec: SASRec, gamma: float = 3.0, fallback=(), n: int = 100):
        self.ease, self.sasrec, self.gamma, self.fallback, self.n = ease, sasrec, gamma, fallback, n

    def ranking(self, library, n: int) -> list[int]:
        cands = self.ease.ranking(library, max(n, self.n))
        ordered = sorted(library, key=lambda a: library[a])  # least recent first
        rank = {a: i for i, a in enumerate(self.sasrec.ranking(ordered, 100))}
        score = {a: 1 - i / len(cands) + self.gamma * (1 - rank[a] / 100 if a in rank else 0.0)
                 for i, a in enumerate(cands)}
        return sorted(cands, key=lambda a: -score[a])[:n]

    def recommend(self, library, k: int = 10) -> list[int]:
        return top_up(self.ranking(library, k), self.fallback, library, k)
