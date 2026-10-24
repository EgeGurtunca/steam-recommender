"""The ranker decides, gemma4 explains. Reranking cost accuracy (see the README), so here ALS keeps its order
and the model only writes one sentence per pick, constrained by a schema to the picks it was given."""
from recs import llm
from recs.core import Library
from recs.data import Catalog
from recs.freeform import normalize
from recs.rerank import OWNED_SHOWN, describe


def schema(picks: list[int]) -> dict:
    item = {"type": "object", "properties": {"app_id": {"type": "integer", "enum": picks},
                                             "reason": {"type": "string"}}, "required": ["app_id", "reason"]}
    return {"type": "object", "properties": {"reasons": {"type": "array", "items": item, "minItems": len(picks),
                                                         "maxItems": len(picks)}}, "required": ["reasons"]}


def grounded(reason: str, library: Library, catalog: Catalog) -> bool:
    """Does the sentence name at least one game the player actually played?"""
    text = f" {normalize(reason)} "
    return any(f" {normalize(catalog.titles[a])} " in text for a in library if a in catalog.titles)


class Explainer:
    def __init__(self, ranker, catalog: Catalog, generate=llm.generate_json):
        self.ranker, self.catalog, self.generate = ranker, catalog, generate
        self.name = f"{ranker.name}+reasons" if hasattr(ranker, "name") else "reasons"
        self.failures = 0

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return self.ranker.recommend(library, k)

    def explain(self, library: Library, k: int = 10) -> list[tuple[int, str]]:
        picks = self.ranker.recommend(library, k)
        owned = sorted(library.items(), key=lambda x: -x[1])[:OWNED_SHOWN]
        played = "\n".join(f"- {describe(a, self.catalog)}: {h:.0f} h" for a, h in owned)
        listed = "\n".join(f"- {a}: {describe(a, self.catalog)}" for a in picks)
        prompt = (f"A Steam player liked these games (hours played):\n{played}\n\n"
                  f"A recommender picked these games for them:\n{listed}\n\n"
                  "For each pick, write one short English sentence on why this player may like it. Name the "
                  "played game(s) that make it a good fit, by their exact title. Do not reorder or add games.")
        try:
            got = {r["app_id"]: r.get("reason", "") for r in self.generate(prompt, schema(picks))["reasons"]}
        except Exception:
            self.failures += 1
            got = {}
        return [(a, got.get(a, "")) for a in picks]
