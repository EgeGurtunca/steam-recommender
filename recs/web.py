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
    played = {a: h for a, h in lib.games.items() if h >= 2}
    taste = {}
    for name, library in (("all", played), ("recent", lib.recent)):
        top = profile.taste_profile(library, store, top=TASTE_TAGS, weights=weights)
        taste[name] = [{"tag": t, "share": round(s, 4)} for t, s in top]
    tags = list(dict.fromkeys(t["tag"] for t in taste["all"] + taste["recent"]))
    picks = [a for a in ranker.ranking(played, 60) if a not in lib.games and profile.is_game(a, store)][:10]
    new = profile.new_releases(played, store, weights=weights)
    top_played = sorted(played.items(), key=lambda x: -x[1])[:6]
    return {"games": len(lib.games), "played": len(played), "hours": round(sum(lib.games.values())),
            "taste": taste, "edges": [{"a": a, "b": b, "w": w} for a, b, w in profile.tag_edges(tags, store)],
            "split": profile.social_split(played, store),
            "top_played": [{"title": lib.titles.get(a, str(a)), "hours": h} for a, h in top_played],
            "next": [card(a, store, catalog) for a in picks], "new": [card(a, store, catalog) for a in new]}


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
        if url.path != "/api/profile":
            return self.send(404, b"not found", "text/plain")
        link = urllib.parse.parse_qs(url.query).get("link", [""])[0]
        try:
            lib = steam.load_library(link)
            body = payload(lib, self.state["store"], self.state["ease"], self.state["catalog"], self.state["weights"])
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
