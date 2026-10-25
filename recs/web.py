"""The local page: paste your Steam profile link, see your taste as a bubble graph and what to play next.

Run: python -m recs.web    then open http://127.0.0.1:8000
Standard library only (http.server); the page itself is static/index.html. Binds to 127.0.0.1, not the network.
"""
import json
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from recs import profile, steam
from recs.data import ROOT, Catalog

PAGE = ROOT / "static" / "index.html"
HEADER = "https://cdn.cloudflare.steamstatic.com/steam/apps/{app}/header.jpg"
TASTE_TAGS = 16


def card(app: int, store: dict, catalog: Catalog) -> dict:
    v = store.get(app, {})
    return {"app_id": app, "title": catalog.titles.get(app) or v.get("title", str(app)),
            "year": (v.get("released") or "")[:4], "tags": v.get("tags", [])[:4],
            "header": HEADER.format(app=app), "url": f"https://store.steampowered.com/app/{app}"}


def payload(lib: steam.SteamLibrary, store: dict, ranker, catalog: Catalog, weights: dict | None = None) -> dict:
    weights = weights or profile.idf(store)
    played, picks = next_picks(lib, store, ranker)
    taste = {}
    for name, library in (("all", played), ("recent", lib.recent)):
        top = profile.taste_profile(library, store, top=TASTE_TAGS, weights=weights)
        taste[name] = [{"tag": t, "share": round(s, 4)} for t, s in top]
    tags = list(dict.fromkeys(t["tag"] for t in taste["all"] + taste["recent"]))
    new = profile.new_releases(played, store, weights=weights)
    top_played = sorted(played.items(), key=lambda x: -x[1])[:6]
    return {"games": len(lib.games), "played": len(played), "hours": round(sum(lib.games.values())),
            "taste": taste, "edges": [{"a": a, "b": b, "w": w} for a, b, w in profile.tag_edges(tags, store)],
            "split": profile.social_split(played, store),
            "top_played": [{"title": lib.titles.get(a, str(a)), "hours": h} for a, h in top_played],
            "next": [card(a, store, catalog) for a in picks], "new": [card(a, store, catalog) for a in new]}


def next_picks(lib: steam.SteamLibrary, store: dict, ranker) -> tuple[dict, list[int]]:
    played = {a: h for a, h in lib.games.items() if h >= 2}
    return played, [a for a in ranker.ranking(played, 60) if a not in lib.games and profile.is_game(a, store)][:10]


def reasons(lib: steam.SteamLibrary, store: dict, ranker, catalog: Catalog, generate=None) -> list[dict]:
    """gemma4's one line per "play next" card, in the page's order (the explainer never reorders)."""
    from recs import explain, llm

    class Fixed:  # the explainer asks its ranker for picks; give it the page's picks
        def recommend(self, library, k=10):
            return picks[:k]

    played, picks = next_picks(lib, store, ranker)
    # names and tags for every game, including the 2023-2026 ones only the store catalogue knows
    merged = Catalog({a: v["title"] for a, v in store.items()} | catalog.titles | lib.titles,
                     {a: v["tags"] for a, v in store.items()} | {a: t for a, t in catalog.tags.items() if t}, {})
    e = explain.Explainer(Fixed(), merged, generate=generate or llm.generate_json)
    return [{"app_id": a, "reason": r} for a, r in e.explain(played, len(picks)) if r]


class Handler(BaseHTTPRequestHandler):
    state: dict = {}

    def send(self, code: int, body: bytes, kind: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        if url.path == "/":
            return self.send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if url.path not in ("/api/profile", "/api/reasons"):
            return self.send(404, b"not found", "text/plain")
        link = urllib.parse.parse_qs(url.query).get("link", [""])[0]
        try:
            lib = steam.load_library(link)
            st = self.state
            if url.path == "/api/reasons":
                from recs import llm
                with llm.ollama():  # started for this request and stopped (with its children) right after
                    body = reasons(lib, st["store"], st["ease"], st["catalog"])
            else:
                body = payload(lib, st["store"], st["ease"], st["catalog"], st["weights"])
            return self.send(200, json.dumps(body).encode(), "application/json")
        except (ValueError, steam.PrivateLibrary) as e:
            return self.send(400, json.dumps({"error": str(e)}).encode(), "application/json")

    def log_message(self, *args):  # quiet: one line per request is noise here
        pass


def main(port: int = 8000) -> None:
    from recs.data import BUILD_DIR, load_catalog, load_store
    from recs.ease import EASE
    print("loading the store catalogue and the EASE weights...")
    store = load_store()
    Handler.state = {"store": store, "catalog": load_catalog(), "ease": EASE.load(BUILD_DIR / "ease"),
                     "weights": profile.idf(store)}
    print(f"open http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
