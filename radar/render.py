"""Compose the Telegram Rich Message (Bot API 10.1+ `sendRichMessage`).

Schema notes — all verified live against api.telegram.org, because several
plausible names are rejected:

    OK              REJECTED
    heading         section_heading, sectionHeading, title, h1
    pre             preformatted, code
    blockquote      block_quotation, quotation
    expandable_blockquote   expandableBlockQuotation
    mathematical_expression math, latex, equation
    voice_note      voiceNote

`heading` additionally requires an integer `size` (1 largest … 6 smallest);
omitting it fails with `Can't find field "size"`. `details` needs `summary`
(not `title`), and list items wrap their content in `blocks`.

Nesting: `details` inside `details` renders as collapsible sub-sections, which
is the structure this channel is built around. Checklists are list items with
`has_checkbox`.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from . import config
from .sources import Story

# ── RichText helpers ─────────────────────────────────────────────────────
def bold(text):    return {"type": "bold", "text": text}
def italic(text):  return {"type": "italic", "text": text}
def code(text):    return {"type": "code", "text": text}
def marked(text):  return {"type": "marked", "text": text}
def spoiler(text): return {"type": "spoiler", "text": text}
def link(text, url): return {"type": "url", "text": text, "url": url}


# ── block helpers ────────────────────────────────────────────────────────
def heading(text, size=2):  return {"type": "heading", "text": text, "size": size}
def para(text):             return {"type": "paragraph", "text": text}
def footer(text):           return {"type": "footer", "text": text}
def divider():              return {"type": "divider"}
def math(expr):             return {"type": "mathematical_expression", "expression": expr}


def photo(url, caption=None):
    """A photo block.

    Schema probed live against api.telegram.org, because every plausible
    shorthand is rejected:

        {"type": "photo", "photo": "<url>"}            -> Field "photo" must be of type Object
        {"type": "photo", "photo": {"url": "<url>"}}   -> Can't find field "type"
        {"type": "photo", "photo": {"type": "photo", "url": ...}}  -> media not found

    The inner object is an InputMedia, so the key is ``media``. Telegram
    downloads the URL server-side and answers with three ready-made thumbnail
    sizes. ``caption`` must be an object (a bare string fails with
    "RichBlockCaption must be an object") and accepts nested rich entities.
    """
    block = {"type": "photo", "photo": {"type": "photo", "media": url}}
    if caption:
        block["caption"] = caption if isinstance(caption, dict) else {"text": caption}
    return block


def details(summary, blocks, is_open=False):
    block = {"type": "details", "summary": summary, "blocks": blocks}
    if is_open:
        block["is_open"] = True
    return block


def checklist(items):
    """items: [(text, checked)] -> a list block with checkboxes."""
    return {"type": "list", "items": [
        {"blocks": [para(text)], "has_checkbox": True, **({"is_checked": True} if done else {})}
        for text, done in items
    ]}


def numbered(items):
    return {"type": "list", "items": [
        {"blocks": [para(text)], "value": i, "label_type": "1"}
        for i, text in enumerate(items, 1)
    ]}


def bullets(items):
    return {"type": "list", "items": [{"blocks": [para(t)]} for t in items]}


def quote(text, credit=None):
    block = {"type": "expandable_blockquote", "text": text}
    if credit:
        block["credit"] = credit
    return block


# Telegram's server-side linkifier turns anything shaped like `host.tld` into a
# clickable URL *after* the payload is accepted — the bot never sees it coming.
# Live post #98 shipped a cell reading "arXiv cs.AI" that Telegram rewrote into
# a link pointing at the literal string "cs.AI", i.e. a dead link on a feed
# category name. Verified by probe: ".AI" is a real ccTLD (Anguilla) so it fires,
# while "cs.LG" is left alone because .LG is not a TLD — which is why only one of
# three arXiv rows looked broken and the bug survived earlier review.
#
# U+2060 WORD JOINER breaks the hostname pattern while being zero-width and
# non-breaking, so the text still reads "cs.AI" to a human. Probed alternatives:
# ZWNJ (U+200C) also works but is a *semantic* character in Persian typing and
# has no business inside a Latin token; a plain space changes the text.
_WORD_JOINER = "\u2060"
_LOOKS_LIKE_HOST = re.compile(r"(?<=[A-Za-z0-9])\.(?=[A-Za-z]{2,24}\b)")


def no_autolink(value):
    """Neutralise accidental host-shaped runs anywhere in a payload subtree.

    Recurses through every container the schema uses (``blocks``, ``items``,
    ``cells``, ``summary``, ``caption`` …) so one call at the end of ``build``
    covers the whole bulletin. A deliberate ``url`` element is returned
    untouched: its label is allowed to read like a hostname.
    """
    if isinstance(value, str):
        return _LOOKS_LIKE_HOST.sub("." + _WORD_JOINER, value)
    if isinstance(value, list):
        return [no_autolink(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "url":
            return value
        return {k: (v if k in _RAW_KEYS else no_autolink(v)) for k, v in value.items()}
    return value


# Keys whose values are machine-read, not displayed prose.
_RAW_KEYS = {"type", "url", "align", "valign", "label_type", "size", "value",
             "is_open", "is_header", "is_bordered", "is_striped", "is_compact",
             "has_checkbox", "is_checked", "is_rtl", "label", "media", "photo",
             "language", "has_spoiler"}


def table(rows, *, header=True, caption=None):
    cells = []
    for r, row in enumerate(rows):
        line = []
        for value in row:
            cell = {"text": value, "align": "center"}
            if header and r == 0:
                cell["is_header"] = True
            line.append(cell)
        cells.append(line)
    block = {"type": "table", "cells": cells, "is_bordered": True,
             "is_striped": True, "is_compact": True}
    if caption:
        block["caption"] = caption
    return block


# ── date line ────────────────────────────────────────────────────────────
_FA_MONTHS = ["فروردین", "اردیبهشت", "خرداد", "تیر", "مرداد", "شهریور",
              "مهر", "آبان", "آذر", "دی", "بهمن", "اسفند"]
_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def _jalali(dt: datetime) -> str:
    """Gregorian → Jalali (Birashk-style civil algorithm, no dependency)."""
    gy, gm, gd = dt.year, dt.month, dt.day
    g_d_m = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    gy2 = gy - 1600
    days = (365 * gy2 + (gy2 + 3) // 4 - (gy2 + 99) // 100 + (gy2 + 399) // 400
            + g_d_m[gm - 1] + gd - 1)
    if gm > 2 and ((gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0):
        days += 1
    days -= 79
    j_np = days // 12053
    days %= 12053
    jy = 979 + 33 * j_np + 4 * (days // 1461)
    days %= 1461
    if days >= 366:
        jy += (days - 1) // 365
        days = (days - 1) % 365
    if days < 186:
        jm = 1 + days // 31
        jd = 1 + days % 31
    else:
        jm = 7 + (days - 186) // 30
        jd = 1 + (days - 186) % 30
    return f"{jd} {_FA_MONTHS[jm - 1]} {jy}".translate(_DIGITS)


def _tehran_now() -> datetime:
    return datetime.now(timezone.utc) + timedelta(hours=3, minutes=30)


# ── the bulletin ─────────────────────────────────────────────────────────
def build(stories: list[Story], summary_fa: str = "") -> dict:
    now = _tehran_now()
    clock = f"{now.hour:02d}:{now.minute:02d}".translate(_DIGITS)

    blocks: list[dict] = [
        heading("رادار هوش مصنوعی", size=1),
        para([italic(f"{_jalali(now)} — ساعت {clock} به وقت تهران"),
              "  •  ", code(f"{len(stories)}".translate(_DIGITS)), " خبر منتخب"]),
    ]

    # Lead image: the best picture of the highest-scoring story, shown before the
    # fold so the bulletin has a face in the channel feed instead of a wall of
    # text. Telegram renders a `photo` block full-width at the top.
    lead = next((s for s in stories if s.image), None)
    if lead is not None:
        blocks.append(photo(lead.image, caption={"text": [
            bold("تصویر شاخص: "), lead.title_fa]}))

    if summary_fa:
        blocks.append(quote(summary_fa, credit="جمع‌بندی سردبیر"))

    blocks.append(_glance(stories))
    blocks.append(divider())

    # Top-level toggle per section, each holding one nested toggle per story.
    by_section: dict[str, list[Story]] = {}
    for s in stories:
        by_section.setdefault(s.section, []).append(s)

    first = True
    for key, label in config.SECTIONS:
        group = by_section.get(key) or []
        if not group:
            continue
        inner: list[dict] = []
        for i, s in enumerate(group, 1):
            inner.append(details(
                [code(f"{i}".translate(_DIGITS)), " ", bold(s.title_fa)],
                _story_blocks(s, with_image=s is not lead),
            ))
        count = f"{len(group)}".translate(_DIGITS)
        blocks.append(details([bold(label), "  ", italic(f"({count} خبر)")],
                              inner, is_open=first))
        first = False

    blocks.append(divider())
    blocks.append(details(bold("🗂 فهرست منابع این بولتن"),
                          [bullets(sorted({f"{s.source} — {s.source_fa}" for s in stories}))]))

    blocks.append(footer([
        "رادار هوش مصنوعی  •  ",
        link("@ai_newsBY", "https://t.me/ai_newsBY"),
        "  •  به‌روزرسانی هر ۶ ساعت",
    ]))
    # Applied once over the finished tree: any host-shaped run that slipped in
    # from a feed title, a category name or a benchmark label would otherwise be
    # silently rewritten into a dead link by Telegram after delivery.
    return {"blocks": no_autolink(blocks), "is_rtl": True}


def _glance(stories: list[Story]) -> dict:
    """A one-line-per-story index, so the reader can decide what to open.

    Typography carries the hierarchy here, since the API has no font field:
    ``superscript`` for the rank, ``bold`` for the headline, ``code`` for the
    score. Probed live — ``subscript``/``superscript``/``marked``/``underline``
    are all accepted as rich-text entities, while ``small``, ``big`` and
    ``highlight`` are rejected.
    """
    rows = []
    for i, s in enumerate(stories, 1):
        rows.append([{"type": "superscript", "text": f"{i}".translate(_DIGITS)},
                     " ", bold(s.title_fa), " ",
                     {"type": "marked", "text": f" {s.score:.0f}".translate(_DIGITS) + "/۱۰ "}])
    return details([bold("⚡️ یک نگاه سریع"), "  ", italic("(فهرست تیترها)")],
                   [numbered_rich(rows)], is_open=True)


def numbered_rich(rows) -> dict:
    """Numbered list whose items carry rich text rather than a plain string."""
    return {"type": "list", "items": [
        {"blocks": [para(r)], "value": i, "label_type": "1"}
        for i, r in enumerate(rows, 1)
    ]}


def _story_blocks(s: Story, with_image: bool = True) -> list[dict]:
    out: list[dict] = [para(s.summary_fa)]
    # Article art goes right after the lede, inside the story's own toggle, so a
    # bulletin of nine stories stays a compact wall of headlines until opened —
    # then each one unfolds with its own picture. The story already used as the
    # lead image skips it so the same photo never appears twice.
    if s.image and with_image:
        out.append(photo(s.image, caption={"text": [
            italic("تصویر: "), s.source_fa]}))
    if s.why_fa:
        out.append(quote(s.why_fa, credit="چرا مهم است"))
    if s.facts:
        out.append(details(italic("نکات کلیدی"), [checklist([(f, True) for f in s.facts])]))
    rows = [["منبع", "اهمیت", "زمان"],
            [s.source, f"{s.score:.0f}".translate(_DIGITS) + "/۱۰",
             _tehran_time_of(s.published)]]
    out.append(table(rows, caption=italic("مشخصات خبر")))
    if s.also_seen_in:
        out.append(para([italic("پوشش خبری دیگر: "), ", ".join(s.also_seen_in)]))
    out.append(para(link("خواندن متن کامل در منبع ↗", s.url)))
    return out


def _tehran_time_of(dt: datetime) -> str:
    local = dt.astimezone(timezone.utc) + timedelta(hours=3, minutes=30)
    return f"{local.hour:02d}:{local.minute:02d}".translate(_DIGITS)
