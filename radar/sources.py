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
            published=published, summary_en=summary,
        ))
    return out


def fetch_article(url: str, limit: int = 2500) -> str:
    """Pull the readable body of an article.

    Vendor blogs (OpenAI, DeepMind, Mistral) frequently ship an empty or
    one-line RSS <description>, which starves the translator: it can only
    restate the headline, so the "key facts" degenerate into filler like
    "published by DeepMind". Fetching the page gives the model real material.

    Best-effort by design — any failure returns "" and the caller falls back to
    the feed summary.
    """
    try:
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
        raw = urllib.request.urlopen(req, timeout=20).read()
    except Exception:
        return ""

    try:
        text = raw.decode("utf-8", "replace")
    except Exception:
        return ""

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
    return " ".join(parts)[:limit]


_STOP = {"the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
         "is", "are", "was", "its", "it", "as", "at", "by", "from", "new",
         "now", "has", "have", "will", "that", "this", "you", "your", "how",
         "why", "what", "says", "said", "can", "could", "more", "than", "but"}


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
                    "Import AI", "Google AI"}


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
