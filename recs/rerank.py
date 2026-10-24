"""gemma4 reorders the hybrid's 50 candidates and says why. The JSON schema only allows candidate ids, so it
cannot recommend a game that does not exist or that is not in the list; anything odd is replaced from the
hybrid order, and a failed call falls back to the hybrid order and is counted."""
from recs import llm
from recs.core import Library, top_up
from recs.data import Catalog

OWNED_SHOWN, TAGS_SHOWN = 25, 8


def schema(candidates: list[int], k: int) -> dict:
    pick = {"type": "object", "properties": {"app_id": {"type": "integer", "enum": candidates},
                                             "reason": {"type": "string"}}, "required": ["app_id", "reason"]}
    return {"type": "object", "properties": {"picks": {"type": "array", "items": pick, "minItems": k, "maxItems": k}},
            "required": ["picks"]}


def describe(app: int, catalog: Catalog) -> str:
    return f"{catalog.titles.get(app, app)} [{', '.join(catalog.tags.get(app, [])[:TAGS_SHOWN])}]"


def build_prompt(library: Library, candidates: list[int], catalog: Catalog, k: int) -> str:
    owned = sorted(library.items(), key=lambda x: -x[1])[:OWNED_SHOWN]
    played = "\n".join(f"- {describe(a, catalog)}: {h:.0f} h" for a, h in owned)
    options = "\n".join(f"- {a}: {describe(a, catalog)}" for a in candidates)
    return (f"A Steam player liked these games (hours played):\n{played}\n\n"
            "The candidates below are already ranked, best first, by a recommender trained on 1.5 million players. "
            "Keep that order unless the games this player played most give you a clear reason to move a game. "
            f"Pick the {k} games this player is most likely to enjoy next, best first. "
            "Use the games they played most as the main evidence. For each pick, write one short English sentence "
            "that names the played game(s) behind it.\n\n"
            f"Candidates (app_id: title [tags]):\n{options}\n")


class LLMRerank:
    name = "hybrid+gemma4"

    def __init__(self, hybrid, catalog: Catalog, generate=llm.generate_json, name: str = "hybrid+gemma4"):
        self.hybrid, self.catalog, self.generate, self.name = hybrid, catalog, generate, name
        self.failures = 0

    def recommend_with_reasons(self, library: Library, k: int = 10) -> list[tuple[int, str]]:
        cands = self.hybrid.candidates(library)
        try:
            picks = self.generate(build_prompt(library, cands, self.catalog, k), schema(cands, k))["picks"]
        except Exception:
            self.failures += 1
            picks = []
        allowed = set(cands)
        reasons = {}
        for p in picks:
            if p.get("app_id") in allowed and p["app_id"] not in reasons:
                reasons[p["app_id"]] = p.get("reason", "")
        ids = top_up(list(reasons), cands, library, k)
        return [(a, reasons.get(a, "")) for a in ids]

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        return [a for a, _ in self.recommend_with_reasons(library, k)]
