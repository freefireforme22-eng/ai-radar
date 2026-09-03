"""Fetch and normalise AI news from RSS/Atom feeds."""
from __future__ import annotations

import hashlib
import html
import re
import time
import urllib.request
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime

from . import config

ATOM = "{http://www.w3.org/2005/Atom}"
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


@dataclass
class Story:
    title_en: str
    url: str
    source: str
    source_fa: str
    tier: int
    published: datetime
    summary_en: str = ""
    # filled in by the enrich stage
    title_fa: str = ""
    summary_fa: str = ""
    why_fa: str = ""
    section: str = "models"
    score: float = 0.0
    facts: list[str] = field(default_factory=list)
    also_seen_in: list[str] = field(default_factory=list)
    image: str = ""

    @property
    def fingerprint(self) -> str:
        base = _norm_url(self.url) or self.title_en.lower()
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]


def _norm_url(url: str) -> str:
    url = (url or "").strip()
    url = re.sub(r"[?#].*$", "", url)
    url = re.sub(r"^https?://(www\.)?", "", url)
    return url.rstrip("/").lower()


def clean_text(raw: str | None, limit: int = 900) -> str:
    if not raw:
        return ""
    txt = _TAG_RE.sub(" ", raw)
    # html.unescape handles numeric entities too — feeds emit &#8217; (curly
    # apostrophe) and a hand-rolled replace() table silently leaves them in the
    # headline, which then reaches the translator as literal "&#8217;".
    txt = html.unescape(txt)
    txt = _WS_RE.sub(" ", txt).strip()
    return txt[:limit]


def _parse_date(text: str | None) -> datetime:
    if not text:
        return datetime.now(timezone.utc)
    text = text.strip()
    try:
        dt = parsedate_to_datetime(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"):
        try:
            dt = datetime.strptime(text.replace("Z", "+0000"), fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except Exception:
            continue
    return datetime.now(timezone.utc)


def _text(node, *names) -> str:
    for n in names:
        el = node.find(n)
        if el is not None:
            if el.text:
                return el.text
            # Atom <content> can carry markup children
            inner = "".join(el.itertext())
            if inner:
                return inner
    return ""


def _link(node) -> str:
    el = node.find("link")
    if el is not None and el.text:
        return el.text.strip()
    for el in node.findall(f"{ATOM}link"):
        rel = el.get("rel", "alternate")
        if rel == "alternate" and el.get("href"):
            return el.get("href").strip()
    el = node.find(f"{ATOM}id")
    if el is not None and el.text and el.text.startswith("http"):
        return el.text.strip()
    return ""


_MEDIA = "{http://search.yahoo.com/mrss/}"

# Tracking pixels, share buttons, and site logos that masquerade as article art.
# `logo` has to match anywhere in the path, not just immediately before the
# extension: a real dry-run illustrated two arXiv papers with
# `static.arxiv.org/icons/twitter/arxiv-logo-twitter-square.png`, which is the
# site's own Twitter card badge. A generic site logo on every paper is worse
# than no picture, because it makes distinct stories look like duplicates.
_JUNK_IMAGE = re.compile(
    r"(?i)(doubleclick|scorecardresearch|google-analytics|/pixel|spacer|"
    r"1x1|blank\.gif|gravatar|feedburner|/badge|/button|logo|/icons?/|"
    r"favicon|placeholder|default-?(image|thumb))"
)


def _image_of(node) -> str:
    """Best image shipped alongside a feed item, or "".

    Probed live across all 22 feeds: 9 carry an image inline, and they use four
    different carriers — <enclosure>, media:content, media:thumbnail, and a bare
    <img> inside the HTML description. Checking only one of them (the usual
    mistake) finds art for Wired but not VentureBeat, or the reverse.
    """
    for tag, attr in (("enclosure", "url"),
                      (_MEDIA + "content", "url"),
                      (_MEDIA + "thumbnail", "url")):
        for el in node.findall(tag):
            url = el.get(attr) or ""
            kind = (el.get("type") or "") + (el.get("medium") or "")
            if url.startswith("http") and ("image" in kind or tag.endswith("thumbnail")):
                if not _JUNK_IMAGE.search(url):
                    return url
    # <img> embedded in description/content HTML (Meta, The Verge, Simon Willison)
    blob = "".join(node.itertext())
    for m in re.finditer(r'<img[^>]+src="(https?://[^"]+)"', blob):
        if not _JUNK_IMAGE.search(m.group(1)):
            return m.group(1)
    return ""


def fetch_feed(feed: dict, cutoff: datetime) -> list[Story]:
    try:
        req = urllib.request.Request(feed["url"], headers={"User-Agent": config.USER_AGENT})
        raw = urllib.request.urlopen(req, timeout=25).read()
        root = ET.fromstring(raw)
    except Exception:
        return []

    nodes = root.findall(".//item") or root.findall(f".//{ATOM}entry")
    out: list[Story] = []
    for node in nodes[:40]:
        title = clean_text(_text(node, "title", f"{ATOM}title"), 300)
        url = _link(node)
        if not title or not url:
            continue
        published = _parse_date(
            _text(node, "pubDate", "published", f"{ATOM}published", f"{ATOM}updated")
            or _text(node, "{http://purl.org/dc/elements/1.1/}date")
        )
        # arXiv feeds republish the whole day's list with today's date; the
        # lookback window still keeps volume sane because we cap per feed.
        if published < cutoff:
            continue
        summary = clean_text(
            _text(node, "description", f"{ATOM}summary", f"{ATOM}content",
                  "{http://purl.org/rss/1.0/modules/content/}encoded"),
            1200,
        )
        out.append(Story(
            title_en=title, url=url, source=feed["name"],
            source_fa=feed.get("fa", feed["name"]), tier=feed.get("tier", 2),
            published=published, summary_en=summary, image=_image_of(node),
        ))
    return out


_OG_IMAGE = (
    r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image',
    r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
)


def _og_image_from(html: str, page_url: str) -> str:
    """og:image of an already-downloaded page, or "".

    Probed live: 9 of 22 feeds ship an image inline, and og:image recovers art
    for 8 of the remaining 13 (DeepMind, Hugging Face, Mistral, Microsoft, MIT
    Tech Review, TechCrunch, Stratechery, Sequoia). arXiv only offers its own
    site logo, which is worse than no picture, so relative URLs are dropped
    rather than resolved.
    """
    for pattern in _OG_IMAGE:
        m = re.search(pattern, html, re.I)
        if not m:
            continue
        url = m.group(1).strip()
        if url.startswith("//"):
            url = "https:" + url
        if url.startswith("http") and not _JUNK_IMAGE.search(url):
            return url
    return ""


def fetch_article(url: str, limit: int = 2500) -> str:
    """Readable body of an article (see :func:`fetch_article_and_image`)."""
    return fetch_article_and_image(url, limit)[0]


def fetch_article_and_image(url: str, limit: int = 2500) -> tuple[str, str]:
    """Pull the readable body of an article.

    Vendor blogs (OpenAI, DeepMind, Mistral) frequently ship an empty or
    one-line RSS <description>, which starves the translator: it can only
    restate the headline, so the "key facts" degenerate into filler like
    "published by DeepMind". Fetching the page gives the model real material.

    Best-effort by design — any failure returns ("", "") and the caller falls
    back to the feed summary. The og:image is harvested from the SAME response,
    so illustrating a story costs no extra HTTP request.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
        raw = urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        return "", ""

    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return "", ""

    image = _og_image_from(text, url)

    # Drop the parts of the DOM that never contain article prose.
    text = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form|noscript)[^>]*>.*?</\1>",
                  " ", text)

    body = re.search(r"(?is)<article[^>]*>(.*?)</article>", text)
    chunk = body.group(1) if body else text

    paragraphs = re.findall(r"(?is)<p[^>]*>(.*?)</p>", chunk)
    parts: list[str] = []
    for p in paragraphs:
        cleaned = clean_text(p, 600)
        # Skip nav/legal boilerplate: cookie banners, share prompts, bylines.
        if len(cleaned) < 60:
            continue
        low = cleaned.lower()
        if any(bad in low for bad in ("cookie", "subscribe", "sign up", "all rights reserved",
                                      "privacy policy", "terms of service", "follow us")):
            continue
        parts.append(cleaned)
        if sum(len(x) for x in parts) > limit:
            break
    return " ".join(parts)[:limit], image


_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
         "is", "are", "was", "its", "it", "as", "at", "by", "from", "new",
         "now", "has", "have", "will", "that", "this", "you", "your", "how",
         "why", "what", "says", "said", "can", "could", "more", "than", "but"}


# ── sources with no usable feed ───────────────────────────────────────────
# Anthropic is a tier-1 AI lab with no RSS at all: every documented endpoint
# (news.rss, rss.xml, feed.xml, atom.xml, /news/index.xml) returns 404, the
# RSSHub mirror answers 403, and openrss.org serves an HTML page rather than a
# feed. Verified by probing all of them. Skipping Anthropic means Claude
# launches never reach a Persian AI-news channel, so its index page is parsed
# directly. The index conveniently carries a date next to every headline, which
# is what makes the lookback window still work.
SCRAPE_SOURCES = [
    {
        "name": "Anthropic",
        "fa": "آنتروپیک",
        "tier": 1,
        "index": "https://www.anthropic.com/news",
        "base": "https://www.anthropic.com",
        "link_re": r'href="(/news/[a-z0-9\-]+)"(.{0,900}?)</a>',
        "date_re": r"([A-Z][a-z]{2} \d{1,2}, \d{4})",
    },
]

_MONTHS = {m: i for i, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], start=1)}

# Category chips that sit inside the same anchor as the headline.
_SCRAPE_LABELS = {"Announcements", "Product", "Policy", "Research",
                  "Societal Impacts", "Interpretability", "Alignment"}


def _scrape_date(blob: str, pattern: str) -> datetime | None:
    hit = re.search(pattern, blob)
    if not hit:
        return None
    try:
        month, day, year = re.match(r"([A-Z][a-z]{2}) (\d{1,2}), (\d{4})",
                                    hit.group(1)).groups()
        return datetime(int(year), _MONTHS[month], int(day), tzinfo=timezone.utc)
    except (AttributeError, KeyError, ValueError):
        return None


def fetch_scraped(cutoff: datetime) -> list[Story]:
    """Stories from sources that publish no feed at all.

    Best-effort like fetch_feed: any failure yields nothing rather than raising,
    because one unreachable site must never stop a bulletin from going out.
    """
    out: list[Story] = []
    for src in SCRAPE_SOURCES:
        try:
            req = urllib.request.Request(src["index"],
                                         headers={"User-Agent": config.USER_AGENT})
            page = urllib.request.urlopen(req, timeout=25).read().decode("utf-8", "replace")
        except Exception:
            continue

        seen_slugs: set[str] = set()
        for match in re.finditer(src["link_re"], page, re.S):
            slug, blob = match.group(1), match.group(2)
            if slug in seen_slugs:
                continue
            seen_slugs.add(slug)
            published = _scrape_date(re.sub(r"<[^>]+>", " ", blob), src["date_re"])
            if published is None or published < cutoff:
                continue
            # The anchor's own opening tag is still attached to the blob (the
            # regex captures from just after the href value), so its class
            # attribute would otherwise be mistaken for text. Drop it first.
            inner = blob.split(">", 1)[1] if ">" in blob else blob
            fragments = [clean_text(f, 300) for f in re.split(r"<[^>]+>", inner)]
            fragments = [f for f in fragments
                         if len(f) > 8
                         and not re.fullmatch(src["date_re"], f)
                         and f not in _SCRAPE_LABELS]
            if not fragments:
                continue
            # Layout is category / date / headline / teaser, and the date and
            # category are now gone, so the headline leads and the rest is body.
            title, teaser = fragments[0], " ".join(fragments[1:])[:600]
            out.append(Story(
                title_en=title, url=src["base"] + slug, source=src["name"],
                source_fa=src.get("fa", src["name"]), tier=src.get("tier", 2),
                published=published, summary_en=teaser,
            ))
    return out


def _shingle(title: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", title.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


# ── AI relevance ─────────────────────────────────────────────────────────
# General-interest feeds (Hacker News, Microsoft Source, Ars) carry plenty of
# non-AI material: "Explore the Designed for Xbox Cozy Collection",
# "Paint.net 5.2 alpha now runs on Linux", "Biggest dark matter detector...".
# Letting those reach the LLM wastes tokens and, worse, they occasionally score
# well and end up in an AI bulletin. Filter before triage, not after.
_AI_TERMS = (
    "ai", "a.i", "artificial intelligence", "machine learning", "deep learning",
    "neural", "llm", "large language model", "chatbot", "generative",
    "transformer", "diffusion", "gpt", "chatgpt", "claude", "gemini", "llama",
    "mistral", "qwen", "deepseek", "grok", "copilot", "openai", "anthropic",
    "deepmind", "hugging face", "huggingface", "midjourney", "stable diffusion",
    "sora", "agentic", "ai agent", "inference", "fine-tune", "fine tune",
    "rag", "embedding", "multimodal", "benchmark", "gpu", "tpu", "nvidia",
    "training data", "model weights", "open-weight", "alignment", "agi",
    "prompt", "tokenizer", "reasoning model", "superintelligence",
)
# Feeds that are AI-only by construction: skip the keyword test for them.
_TRUSTED_TOPICAL = {"OpenAI", "DeepMind", "Hugging Face", "Mistral",
                    "arXiv cs.AI", "arXiv cs.LG", "arXiv cs.CL",
                    "Import AI", "Google AI", "Anthropic"}


def is_ai_related(story: Story) -> bool:
    if story.source in _TRUSTED_TOPICAL:
        return True
    haystack = f" {story.title_en.lower()} {story.summary_en[:400].lower()} "
    for term in _AI_TERMS:
        if term == "ai":
            # bare "AI" must be a standalone token, else "said"/"maintain" match
            if re.search(r"(?<![a-z])ai(?![a-z])", haystack):
                return True
            continue
        if term in haystack:
            return True
    return False


def dedupe_similar(stories: list[Story], threshold: float = 0.42) -> list[Story]:
    """Collapse the same event reported by several outlets.

    URL fingerprints miss this entirely: a Gemini launch shows up from Google's
    own blog, TechCrunch and The Verge with three different URLs. Two signals
    are used together because either alone leaks duplicates:

      * Jaccard overlap on title keywords (catches reworded headlines);
      * containment — when the shorter headline's keywords are ~fully inside the
        longer one, e.g. "Trump Administration Sides With OpenAI in New York
        Times Copyright Lawsuit" vs "The Trump administration is supporting
        OpenAI in the NYT copyright lawsuit" (Jaccard only 0.50).

    The lowest-tier (most primary) source wins, so the vendor announcement is
    what gets published and the aggregators are credited instead.
    """
    kept: list[tuple[set[str], Story]] = []
    for s in sorted(stories, key=lambda x: (x.tier, -x.published.timestamp())):
        sig = _shingle(s.title_en)
        if not sig:
            continue
        duplicate = False
        for other_sig, other in kept:
            union = sig | other_sig
            if not union:
                continue
            overlap = len(sig & other_sig)
            jaccard = overlap / len(union)
            containment = overlap / min(len(sig), len(other_sig))
            if jaccard >= threshold or (containment >= 0.7 and overlap >= 3):
                if other.source != s.source and s.source not in other.also_seen_in:
                    other.also_seen_in.append(s.source)
                duplicate = True
                break
        if not duplicate:
            kept.append((sig, s))
    return [s for _, s in kept]


def collect(lookback_hours: int | None = None) -> list[Story]:
    hours = lookback_hours or config.LOOKBACK_HOURS
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    with ThreadPoolExecutor(max_workers=10) as ex:
        batches = list(ex.map(lambda f: fetch_feed(f, cutoff), config.FEEDS))
    batches.append(fetch_scraped(cutoff))

    seen: set[str] = set()
    stories: list[Story] = []
    for batch in batches:
        for s in batch:
            fp = s.fingerprint
            if fp in seen:
                continue
            seen.add(fp)
            if not is_ai_related(s):
                continue
            stories.append(s)
    stories = dedupe_similar(stories)
    stories.sort(key=lambda s: (s.tier, -s.published.timestamp()))
    return stories
