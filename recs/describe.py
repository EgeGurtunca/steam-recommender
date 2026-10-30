"""What a game is about, as a vector: bge-m3 over its title and store description, through Ollama. A player is the
hours-weighted mean of their games' vectors, and a game nobody has played yet can still be close to it, which tags
alone did poorly at (eval/run_eval.py --cold-start).

Run: python -m recs.describe     (starts Ollama, embeds the 2022 catalogue's descriptions, writes the cache)
"""
import json
import math
import urllib.request

import numpy as np

from recs.core import Library, top_up
from recs.data import BUILD_DIR, RAW_DIR

CACHE = BUILD_DIR / "descriptions.npz"
MODEL = "bge-m3"


def descriptions(apps: set[int]) -> dict[int, str]:
    """Title and store description for each app in the 2024 metadata (an empty description keeps the title)."""
    import duckdb
    rows = duckdb.execute(f"""
        select g.app_id, g.title, coalesce(m.description, '')
        from read_csv('{RAW_DIR.as_posix()}/games.csv', header = true, types = {{'app_id': 'BIGINT'}}) g
        left join read_json('{RAW_DIR.as_posix()}/games_metadata.json', format = 'newline_delimited',
                            columns = {{'app_id': 'BIGINT', 'description': 'VARCHAR'}}) m using (app_id)""").fetchall()
    return {a: f"{t}. {d}".strip() for a, t, d in rows if a in apps}


def embed(texts: list[str], batch: int = 64) -> np.ndarray:
    from recs.llm import OLLAMA_URL
    out = []
    for i in range(0, len(texts), batch):
        body = json.dumps({"model": MODEL, "input": texts[i:i + batch]}).encode()
        req = urllib.request.Request(OLLAMA_URL + "/api/embed", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            out += json.load(r)["embeddings"]
        if (i // batch) % 20 == 0:
            print(f"PROGRESS {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    v = np.asarray(out, dtype=np.float32)
    return v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-9)


def load() -> dict[int, np.ndarray]:
    f = np.load(CACHE)
    return dict(zip(f["apps"].tolist(), f["vectors"]))


class Described:
    """Cosine between a game's vector and the player's (hours-weighted, sqrt as in recs.profile), optionally times
    log1p(positive reviews) ** beta, as the tags x popularity system does."""

    def __init__(self, vectors: dict, fallback=(), prior: dict | None = None, beta: float = 0.0, name: str = "desc"):
        self.apps = list(vectors)
        self.index = {a: i for i, a in enumerate(self.apps)}
        m = np.asarray([vectors[a] for a in self.apps], dtype=np.float32)
        self.m = m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-9)
        self.boost = np.asarray([math.log1p((prior or {}).get(a, 0)) ** beta for a in self.apps], dtype=np.float32)
        self.fallback, self.name = fallback, name

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        known = [a for a in library if a in self.index]
        if not known:
            return top_up([], self.fallback, library, k)
        w = np.sqrt(np.asarray([max(library[a], 0.0) for a in known], dtype=np.float32)) + 1e-6
        user = w @ self.m[[self.index[a] for a in known]]
        scores = (self.m @ user) * self.boost
        order = np.argsort(-scores)[: k + len(library)]
        return top_up([self.apps[i] for i in order], self.fallback, library, k)


class Bridge:
    """A two-tower bridge for games nobody has played: ALS scores the games it knows, and a game it never saw gets a
    factor vector predicted from its description, by a ridge regression fit on the games ALS knows. Ridge shrinks
    its predictions towards the mean, so they are scaled to the known vectors' mean length to compete in one list."""

    def __init__(self, als, vectors: dict, alpha: float = 1.0, fallback=(), name: str = "bridge"):
        from sklearn.linear_model import Ridge
        known = [a for a in als.index if a in vectors]
        x = np.asarray([vectors[a] for a in known], dtype=np.float32)
        y = np.asarray(als.model.item_factors)[[als.index[a] for a in known]]
        self.new = [a for a in vectors if a not in als.index]
        pred = Ridge(alpha=alpha).fit(x, y).predict(np.asarray([vectors[a] for a in self.new], dtype=np.float32))
        scale = np.linalg.norm(y, axis=1).mean() / max(float(np.linalg.norm(pred, axis=1).mean()), 1e-9)
        self.new_f = (pred * scale).astype(np.float32)
        self.als, self.fallback, self.name = als, fallback, name

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        row = self.als._row(library)
        if row is None:
            return top_up([], self.fallback, library, k)
        user = np.asarray(self.als.model.recalculate_user(0, row)).ravel()
        apps = list(self.als.item_ids) + self.new
        scores = np.concatenate([np.asarray(self.als.model.item_factors) @ user, self.new_f @ user])
        order = np.argsort(-scores)[: k + len(library)]
        return top_up([int(apps[i]) for i in order], self.fallback, library, k)


def main() -> None:
    from recs import llm
    from recs.data import load_catalog
    texts = descriptions(set(load_catalog().titles))
    apps = sorted(texts)
    with llm.ollama():
        vectors = embed([texts[a] for a in apps])
    np.savez_compressed(CACHE, apps=np.asarray(apps), vectors=vectors)
    print(f"wrote {CACHE}: {len(apps)} games, {vectors.shape[1]} dimensions")


if __name__ == "__main__":
    main()
