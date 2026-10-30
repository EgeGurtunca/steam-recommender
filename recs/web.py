"""The local page: paste your Steam profile link, see your taste as a bubble graph and what to play next.

Run: python -m recs.web    then open http://127.0.0.1:8000
Standard library only (http.server); the page itself is static/index.html. Binds to 127.0.0.1, not the network.
"""
import json
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from recs import profile, sequence, steam
from recs.core import diversify
from recs.data import BUILD_DIR, ROOT, Catalog

PAGE = ROOT / "static" / "index.html"
HEADER = "https://cdn.cloudflare.steamstatic.com/steam/apps/{app}/header.jpg"
TASTE_TAGS = 16
CANDIDATES = 60  # EASE's top 60 go to the page, which re-sorts them with the popular/niche and older/newer sliders
DIVERSITY = 0.8  # tag variety in that order (recs.core.diversify): no recall lost on the time split, val or test
# "demo" in the link box: a made-up library, so the page can be tried (and shown) without a Steam key or anyone's profile
DEMO = steam.SteamLibrary("demo", {620: 40.0, 105600: 310.0, 413150: 120.0, 367520: 65.0, 292030: 140.0, 489830: 90.0,
                                   1145360: 55.0, 504230: 25.0, 646570: 180.0, 588650: 45.0, 427520: 260.0,
                                   294100: 200.0, 264710: 50.0, 374320: 70.0, 1245620: 110.0, 1086940: 150.0},
                          recent={1086940: 12.0, 646570: 4.0},
                          # played lately: Baldur's Gate 3 and Slay the Spire, then Elden Ring; the rest years ago
                          last_played={1086940: 1761000000, 646570: 1760900000, 1245620: 1745000000})


def card(app: int, store: dict, catalog: Catalog) -> dict:
    v = store.get(app, {})
    return {"app_id": app, "title": catalog.titles.get(app) or v.get("title", str(app)),
            "year": (v.get("released") or "")[:4], "tags": v.get("tags", [])[:4],
            "positive": v.get("positive", 0),
            "header": HEADER.format(app=app), "url": f"https://store.steampowered.com/app/{app}"}


def payload(lib: steam.SteamLibrary, store: dict, ranker, catalog: Catalog, weights: dict | None = None) -> dict:
    weights = weights or profile.idf(store)
    played, picks = next_picks(lib, store, ranker, CANDIDATES)
    tags = {a: store.get(a, {}).get("tags") or catalog.tags.get(a, []) for a in picks}
    picks = diversify(picks, tags, DIVERSITY, len(picks))
    # "because you played X": the biggest term in EASE's sum for each pick (the page's rankers all start from EASE)
    ease = getattr(ranker, "ease", ranker)
    why = ease.because(played, picks) if hasattr(ease, "because") else {}
    because = {a: {"because": lib.titles.get(b) or catalog.titles.get(b) or store.get(b, {}).get("title", str(b))}
               for a, b in why.items()}
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
            "next": [card(a, store, catalog) | because.get(a, {}) for a in picks],
            "new": [card(a, store, catalog) for a in new]}


def recency(lib: steam.SteamLibrary, played: dict, now: float | None = None) -> dict[int, float]:
    """Each played game weighted by how long ago it was last played, half per sequence.HALF_LIFE days: the order
    the ranker's transitions read. A game Steam has no time for counts as the oldest one; no times at all: all 1."""
    now = now or time.time()
    w = {a: 0.5 ** ((now - lib.last_played[a]) / 86400 / sequence.HALF_LIFE) for a in played if a in lib.last_played}
    oldest = min(w.values(), default=1.0)
    return {a: w.get(a, oldest) for a in played}


def next_picks(lib: steam.SteamLibrary, store: dict, ranker, k: int = 10, now: float | None = None) -> tuple[dict, list[int]]:
    played = {a: h for a, h in lib.games.items() if h >= 2}
    picks = ranker.ranking(recency(lib, played, now), k + 50)
    return played, [a for a in picks if a not in lib.games and profile.is_game(a, store)][:k]


def reasons(lib: steam.SteamLibrary, store: dict, ranker, catalog: Catalog, generate=None,
            apps: list[int] | None = None) -> list[dict]:
    """gemma4's one line per "play next" card, in the page's order (the explainer never reorders). `apps` are the
    cards the page shows after its sliders; only unowned games from the store are taken, at most 10."""
    from recs import explain, llm

    class Fixed:  # the explainer asks its ranker for picks; give it the page's picks
        def recommend(self, library, k=10):
            return picks[:k]

    played, picks = next_picks(lib, store, ranker)
    if apps:
        picks = [a for a in dict.fromkeys(apps) if a in store and a not in lib.games][:10]
    # names and tags for every game, including the 2023-2026 ones only the store catalogue knows
    merged = Catalog({a: v["title"] for a, v in store.items()} | catalog.titles | lib.titles,
                     {a: v["tags"] for a, v in store.items()} | {a: t for a, t in catalog.tags.items() if t}, {})
    e = explain.Explainer(Fixed(), merged, generate=generate or llm.generate_json)
    return [{"app_id": a, "reason": r} for a, r in e.explain(played, len(picks)) if r]


def library(link: str) -> steam.SteamLibrary:
    return DEMO if link.strip() == "demo" else steam.load_library(link)


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
        query = urllib.parse.parse_qs(url.query)
        link = query.get("link", [""])[0]
        try:
            apps = [int(a) for a in query.get("apps", [""])[0].split(",") if a]
            lib = library(link)
            st = self.state
            if url.path == "/api/reasons":
                from recs import llm
                with llm.ollama():  # started for this request and stopped (with its children) right after
                    body = reasons(lib, st["store"], st["ranker"], st["catalog"], apps=apps)
            else:
                body = payload(lib, st["store"], st["ranker"], st["catalog"], st["weights"])
            return self.send(200, json.dumps(body).encode(), "application/json")
        except (ValueError, steam.PrivateLibrary) as e:
            return self.send(400, json.dumps({"error": str(e)}).encode(), "application/json")
        except steam.SteamUnavailable as e:
            return self.send(502, json.dumps({"error": str(e)}).encode(), "application/json")

    def log_message(self, *args):  # quiet: one line per request is noise here
        pass


def page_ranker(ease):
    """EASE + SASRec (numpy) if eval/sasrec.py --page has trained it, else EASE + transitions: the two order-aware
    rankers that beat EASE on the time split's test (38% and 14% more)."""
    if (BUILD_DIR / "sasrec_page.npz").exists():
        from recs.sasrec_np import SASRec, SasrecNudge
        print("ranking with EASE + SASRec (numpy)")
        return SasrecNudge(ease, SASRec(BUILD_DIR / "sasrec_page.npz"))
    print("ranking with EASE + transitions (counted from the raw reviews the first time)")
    markov = sequence.Markov(sequence.load_transitions(ease.index, BUILD_DIR / "transitions.npz"), ease.item_ids,
                             ease.index)
    return sequence.MarkovNudge(ease, markov, sequence.GAMMA)


def main(port: int = 8000) -> None:
    from recs.data import BUILD_DIR, load_catalog, load_store
    from recs.ease import EASE
    print("loading the store catalogue and the EASE weights (trained the first time, about 20 s)...")
    store = load_store()
    Handler.state = {"store": store, "catalog": load_catalog(), "ranker": page_ranker(EASE.load_or_build(BUILD_DIR / "ease")),
                     "weights": profile.idf(store)}
    print(f"open http://127.0.0.1:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()


if __name__ == "__main__":
    main()
