"""Read one person's own Steam library from their profile link, at their request (Steam Web API terms: a
user's data only when that user asks). Needs STEAM_API_KEY in .env; nothing read here is stored."""
import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from recs.data import ROOT

API = "https://api.steampowered.com"


class PrivateLibrary(Exception):
    """The profile hides its game details; Steam returns an empty answer instead of an error."""


@dataclass
class SteamLibrary:
    steamid: str
    games: dict[int, float]  # app_id -> total hours
    recent: dict[int, float] = field(default_factory=dict)  # app_id -> hours in the last two weeks
    titles: dict[int, str] = field(default_factory=dict)
    last_played: dict[int, int] = field(default_factory=dict)  # app_id -> unix time, for games ever played


def parse_profile(link: str) -> tuple[str, str]:
    """("vanity", name) for steamcommunity.com/id/<name>, ("id", steamid64) for /profiles/<id> or a bare id."""
    link = link.strip()
    if re.fullmatch(r"\d{17}", link):
        return "id", link
    m = re.search(r"steamcommunity\.com/(id|profiles)/([^/?#]+)", link)
    if not m:
        raise ValueError(f"not a Steam profile link: {link}")
    return ("vanity" if m.group(1) == "id" else "id"), m.group(2)


class SteamUnavailable(Exception):
    """Steam did not answer (network down, timeout, a 5xx)."""


def api_key() -> str:
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines() if (ROOT / ".env").exists() else []:
        if line.startswith("STEAM_API_KEY="):
            return line.split("=", 1)[1].strip()
    if "STEAM_API_KEY" not in os.environ:
        raise ValueError("no Steam API key: put STEAM_API_KEY=... in .env (or type demo for a made-up library)")
    return os.environ["STEAM_API_KEY"]


def http_json(url: str) -> dict:
    """The URL holds the API key, so no error raised here may include it (urllib's own errors do)."""
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code < 500:  # Steam refuses the request: almost always an id that does not exist
            raise ValueError(f"Steam answered {e.code} for this profile: check the link") from None
        raise SteamUnavailable(f"Steam answered {e.code}, try again in a minute") from None
    except (urllib.error.URLError, TimeoutError):
        raise SteamUnavailable("Steam is not answering, try again in a minute") from None


def load_library(link: str, key: str | None = None, get: Callable[[str], dict] = http_json) -> SteamLibrary:
    key = key or api_key()
    kind, value = parse_profile(link)
    if kind == "vanity":
        res = get(f"{API}/ISteamUser/ResolveVanityURL/v1/?key={key}&vanityurl={urllib.parse.quote(value)}")["response"]
        if res.get("success") != 1:
            raise ValueError(f"no Steam profile named {value}")
        value = res["steamid"]
    owned = get(f"{API}/IPlayerService/GetOwnedGames/v1/?key={key}&steamid={value}"
                "&include_appinfo=1&include_played_free_games=1")["response"]
    if "games" not in owned:
        raise PrivateLibrary("this profile's game details are not public (Profile > Privacy > Game details)")
    games = owned["games"]
    return SteamLibrary(value, {g["appid"]: round(g.get("playtime_forever", 0) / 60, 1) for g in games},
                        {g["appid"]: round(g["playtime_2weeks"] / 60, 1) for g in games if g.get("playtime_2weeks")},
                        {g["appid"]: g.get("name", "") for g in games},
                        {g["appid"]: g["rtime_last_played"] for g in games if g.get("rtime_last_played")})
