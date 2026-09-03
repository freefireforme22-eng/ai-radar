"""Geocoding for the `map` block, and arXiv ids for the `pre` block.

Both exist so posts differ from each other: a policy story anchored to Brussels
gets a map no other story has, an arXiv paper gets a monospace citation card no
other story has.

Nominatim gets a 1 req/s courtesy delay and a hard cap per bulletin. Any
failure returns None and the block is simply omitted.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request

from . import config

_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_MAX_LOOKUPS = 3
_lookups = 0
_cache: dict[str, tuple[float, float] | None] = {}

# Place names worth a map, matched against the English source. Free-text
# geocoding of a headline returns nonsense ("Nvidia headquarters Santa Clara"
# resolved to None while plain "Brussels" resolved fine), so the trigger list is
# explicit: regulatory and datacentre geography, where location IS the story.
_PLACES = [
    (r"\b(European Union|EU|Brussels|European Commission)\b", "Brussels, Belgium", "بروکسل — مقر نهادهای اروپا"),
    (r"\bUK|Britain|London|Ofcom\b", "London, United Kingdom", "لندن"),
    (r"\bWashington|White House|Congress|Senate\b", "Washington, D.C., USA", "واشینگتن"),
    (r"\bBeijing|China's|Chinese government\b", "Beijing, China", "پکین"),
    (r"\bShenzhen\b", "Shenzhen, China", "شنژن"),
    (r"\bTaiwan|TSMC|Hsinchu\b", "Hsinchu, Taiwan", "هسینچو — قطب ساخت تراشه"),
    (r"\bSeoul|Korea|Samsung|SK Hynix\b", "Seoul, South Korea", "سئول"),
    (r"\bTokyo|Japan\b", "Tokyo, Japan", "توکیو"),
    (r"\bIndia|Bengaluru|Bangalore|New Delhi\b", "Bengaluru, India", "بنگلورو"),
    (r"\bSan Francisco|Silicon Valley|Bay Area\b", "San Francisco, California", "سان‌فرانسیسکو"),
    (r"\bAbu Dhabi|UAE|Emirates|G42\b", "Abu Dhabi, United Arab Emirates", "ابوظبی"),
    (r"\bSaudi|Riyadh|Neom\b", "Riyadh, Saudi Arabia", "ریاض"),
    (r"\bParis|France|Mistral\b", "Paris, France", "پاریس"),
    (r"\bBerlin|Germany\b", "Berlin, Germany", "برلین"),
    (r"\bIreland|Dublin\b", "Dublin, Ireland", "دوبلین"),
]


def _geocode(query: str) -> tuple[float, float] | None:
    global _lookups
    if query in _cache:
        return _cache[query]
    if _lookups >= _MAX_LOOKUPS:
        return None
    _lookups += 1
    url = f"{_NOMINATIM}?format=json&limit=1&q={urllib.parse.quote(query)}"
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        hit = (float(data[0]["lat"]), float(data[0]["lon"])) if data else None
    except Exception:
        hit = None
    _cache[query] = hit
    time.sleep(1.1)          # Nominatim asks for max 1 req/s
    return hit


def locate(source_text: str) -> tuple[float, float, str] | None:
    """First recognised place in the story's English text, geocoded.

    Returns (lat, lon, persian_label) or None. Only ONE story per bulletin
    should carry a map — the caller enforces that.
    """
    for pattern, query, label in _PLACES:
        if re.search(pattern, source_text or "", re.I):
            hit = _geocode(query)
            if hit:
                return hit[0], hit[1], label
    return None


def reset() -> None:
    """Clear the per-bulletin lookup budget (used by tests)."""
    global _lookups
    _lookups = 0
    _cache.clear()


# arXiv abs/PDF links carry the paper id; a citation card in monospace is the one
# place a `pre` block genuinely belongs in a news bulletin.
_ARXIV_ID = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})(?:v\d+)?", re.I)


def arxiv_id(url: str) -> str:
    m = _ARXIV_ID.search(url or "")
    return m.group(1) if m else ""


def citation(url: str, title_en: str, source: str) -> str:
    """A BibTeX-ish citation card. Latin on purpose: it is a machine identifier,
    not prose, so the Persian ratio audit must not see it (render passes it
    through `pre`, which the audit skips)."""
    ident = arxiv_id(url)
    if not ident:
        return ""
    year = "20" + ident[:2]
    first = re.sub(r"[^A-Za-z0-9 ]", "", title_en or "").split()
    key = (first[0].lower() if first else "paper") + ident[:4]
    return (f"@misc{{{key},\n"
            f"  title  = {{{(title_en or '').strip()[:90]}}},\n"
            f"  eprint = {{{ident}}},\n"
            f"  year   = {{{year}}},\n"
            f"  note   = {{{source}}}\n"
            f"}}")
