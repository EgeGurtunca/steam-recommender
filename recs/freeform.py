"""The experiment: gemma4 alone, no candidate list. It names games from its own memory; we check how many of
them exist in the catalog, how many the player already owns, and whether they hit the hidden games."""
import difflib
import re
import unicodedata

from recs import llm
from recs.core import Library
from recs.data import Catalog
from recs.rerank import OWNED_SHOWN, describe


def normalize(title: str) -> str:
    title = re.sub(r"[™®©]", "", title)  # first: NFKC would turn ™ into the letters "TM"
    title = unicodedata.normalize("NFKC", title).casefold()
    return re.sub(r"[^0-9a-z]+", " ", title).strip()


def match(titles: list[str], catalog: Catalog) -> list[int | None]:
    """Exact normalized title, else a close match (cutoff 0.9), else None."""
    by_name = {normalize(t): a for a, t in catalog.titles.items()}
    out = []
    for t in titles:
        n = normalize(t)
        close = n if n in by_name else next(iter(difflib.get_close_matches(n, by_name, n=1, cutoff=0.9)), None)
        out.append(by_name[close] if close else None)
    return out


class FreeFormLLM:
    name = "gemma4 alone"

    def __init__(self, catalog: Catalog, generate=llm.generate_json, all_titles: dict[int, str] | None = None):
        """all_titles: every app in the dataset, filtered or not, to tell "not in our catalog" from "not on Steam"."""
        self.catalog, self.generate = catalog, generate
        self.everything = Catalog(all_titles or catalog.titles, {}, {})
        self.last: dict = {}
        self.totals = {"titles": 0, "matched": 0, "owned": 0, "on_steam": 0}
        self.failures = 0

    def recommend_titles(self, library: Library, k: int = 10) -> list[str]:
        owned = sorted(library.items(), key=lambda x: -x[1])[:OWNED_SHOWN]
        played = "\n".join(f"- {describe(a, self.catalog)}: {h:.0f} h" for a, h in owned)
        prompt = (f"A Steam player liked these games (hours played):\n{played}\n\n"
                  f"Recommend {k} other Steam games this player is most likely to enjoy next, best first. "
                  "Give exact Steam store titles only.")
        sch = {"type": "object", "properties": {"titles": {"type": "array", "items": {"type": "string"},
                                                           "minItems": k, "maxItems": k}}, "required": ["titles"]}
        try:
            return self.generate(prompt, sch)["titles"]
        except Exception:
            self.failures += 1
            return []

    def recommend(self, library: Library, k: int = 10) -> list[int]:
        titles = self.recommend_titles(library, k)
        ids = match(titles, self.catalog)
        found = [a for a in ids if a is not None]
        on_steam = sum(a is not None for a in match(titles, self.everything))
        self.last = {"titles": len(titles), "matched": len(found), "owned": sum(a in library for a in found),
                     "on_steam": on_steam}
        for key in self.totals:
            self.totals[key] += self.last[key]
        return list(dict.fromkeys(a for a in found if a not in library))[:k]
