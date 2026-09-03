"""Compose the Telegram Rich Message (Bot API 10.1+ `sendRichMessage`).

The block vocabulary was taken from the official reference (core.telegram.org/
bots/api, the `InputRichBlock*` family) rather than guessed, then each entry was
re-probed live because the docs and the server disagree in places:

    24 block types exist: paragraph heading footer divider details list table
    photo video animation audio voice_note document pre blockquote
    expandable_blockquote pullquote collage slideshow map anchor buttons
    mathematical_expression thinking   (`thinking` is draft-only)

    RichText entities: bold italic underline strikethrough spoiler code marked
    subscript superscript url mention hashtag cashtag bot_command custom_emoji
    anchor_link reference reference_link date_time mathematical_expression

Traps found by probing, each of which silently degrades the message:

* Ordered lists key off ``item["type"]`` ("1", "a", "A", "i", "I") — NOT
  ``label_type``. An unknown field is accepted and dropped, so a numbered list
  sent with ``label_type`` renders as bullets. Verified by forwarding the sent
  message back and reading ``label``: with ``type`` it is "1."/"c."/"iii.",
  with ``label_type`` it is "•".
* There is no colour, theme or font control. ``color``, ``accent_color_id``,
  ``theme``, ``style`` and ``font*`` are all accepted and dropped. The only
  real top-level fields are ``is_rtl`` and ``skip_entity_detection``. Colour
  therefore comes from ``buttons`` (``style``: primary/success/danger/link) and
  from custom emoji.
* ``blockquote`` takes ``blocks``, while ``expandable_blockquote`` and
  ``pullquote`` take ``text``. Sending ``text`` to ``blockquote`` fails with
  RICH_MESSAGE_EMPTY.
* ``skip_entity_detection`` is the supported way to stop Telegram rewriting
  "cs.AI" into a dead link; explicit ``url`` entities still work under it.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from . import config
from .sources import Story

# ── RichText helpers ─────────────────────────────────────────────────────
def bold(text):    return {"type": "bold", "text": text}
def italic(text):  return {"type": "italic", "text": text}
def code(text):    return {"type": "code", "text": text}
def marked(text):  return {"type": "marked", "text": text}
def spoiler(text): return {"type": "spoiler", "text": text}
def under(text):   return {"type": "underline", "text": text}
def sup(text):     return {"type": "superscript", "text": text}
def sub(text):     return {"type": "subscript", "text": text}
def link(text, url): return {"type": "url", "text": text, "url": url}
def inline_math(expr): return {"type": "mathematical_expression", "expression": expr}


def anchor_link(text, name=""):
    """Jump to an ``anchor`` block. Empty name jumps back to the top."""
    return {"type": "anchor_link", "text": text, "anchor_name": name}


# `reference` / `reference_link` are documented but the server rejects them:
# 'type "reference" is unsupported'. Deliberately not wrapped here so nobody
# reaches for them and breaks a bulletin.


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


def numbered(items, kind="1", start=1):
    """A real ordered list.

    ``kind`` maps to the documented ``type`` values: "1" decimal, "a"/"A"
    letters, "i"/"I" Roman numerals. Passing it as ``label_type`` (an earlier
    mistake here) is silently ignored and yields bullets.
    """
    return {"type": "list", "items": [
        {"blocks": [para(t)], "type": kind, "value": i}
        for i, t in enumerate(items, start)
    ]}


def bullets(items):
    return {"type": "list", "items": [{"blocks": [para(t)]} for t in items]}


def anchor(name):
    return {"type": "anchor", "name": name}


def buttons(specs, align="center"):
    """A row of 1-8 coloured buttons.

    ``style`` is the only real colour control in the whole API: "primary"
    (blue), "success" (green), "danger" (red) or "link" (plain — note the
    server drops the field in that case, confirmed by round-trip).
    """
    return {"type": "buttons", "align": align, "buttons": [
        {"text": text, "url": url, **({"style": style} if style else {})}
        for text, url, style in specs
    ]}


def pullquote(text, credit=None):
    """Centred aside — visually distinct from both quote styles."""
    block = {"type": "pullquote", "text": text}
    if credit:
        block["credit"] = credit
    return block


def hard_quote(blocks, credit=None):
    """Non-expandable blockquote. Takes ``blocks``, unlike the other two."""
    block = {"type": "blockquote", "blocks": blocks}
    if credit:
        block["credit"] = credit
    return block


def collage(urls, caption=None):
    block = {"type": "collage", "blocks": [
        {"type": "photo", "photo": {"type": "photo", "media": u}} for u in urls]}
    if caption:
        block["caption"] = {"text": caption}
    return block


def slideshow(urls, caption=None):
    block = {"type": "slideshow", "blocks": [
        {"type": "photo", "photo": {"type": "photo", "media": u}} for u in urls]}
    if caption:
        block["caption"] = {"text": caption}
    return block


def pre(text, language="text"):
    return {"type": "pre", "text": text, "language": language}



def quote(text, credit=None):
    block = {"type": "expandable_blockquote", "text": text}
    if credit:
        block["credit"] = credit
    return block


# Telegram's server-side linkifier rewrites anything shaped like `host.tld` into
# a clickable URL *after* the payload is accepted. Live post #98 shipped a cell
# reading "arXiv cs.AI" that became a link pointing at the literal string
# "cs.AI" — a dead link on a feed category name (".AI" is Anguilla's ccTLD, so
# it fires; "cs.LG" is left alone because .LG is not a TLD, which is why only
# one of three arXiv rows looked broken and the bug survived review).
#
# The documented cure is the top-level `skip_entity_detection` flag, set in
# `build`. Probed live: with it, the stored message carries zero `url` entities
# for "cs.AI و openai.com", while an explicit `url` element still works — so
# real links keep working and accidental ones stop appearing. This replaces an
# earlier U+2060 WORD JOINER hack that polluted the text with invisible
# characters to defeat the pattern.


def table(rows, *, header=True, caption=None, striped=True, compact=True):
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
             "is_striped": striped, "is_compact": compact}
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


# ── per-bulletin identity ────────────────────────────────────────────────
# The complaint was that every post looked identical ("قالب پیامای الان همه‌شون
# شبیه همه"). The API exposes no colour or font control, so identity has to come
# from rotating the *structure*: which quote form carries the digest, which
# button colour leads, which glyphs mark the sections. One theme per 6-hour slot
# means four consecutive bulletins never look the same.
_THEMES = [
    {"mark": "🛰", "accent": "primary", "digest": "pullquote",
     "rule": "▬▬▬▬▬", "glance": "⚡️", "gallery": "collage"},
    {"mark": "🌐", "accent": "success", "digest": "blockquote",
     "rule": "◈ ◈ ◈", "glance": "🎯", "gallery": "slideshow"},
    {"mark": "🧭", "accent": "danger", "digest": "expandable",
     "rule": "━━━━━", "glance": "📌", "gallery": "collage"},
    {"mark": "🔭", "accent": "link", "digest": "pullquote",
     "rule": "✦ ✦ ✦", "glance": "🗞", "gallery": "slideshow"},
]

# Section accents: each section gets its own heading size and quote form so a
# research item reads differently from a funding item even inside one bulletin.
_SECTION_STYLE = {
    "models":   {"size": 3, "quote": "expandable", "bullet": "i"},
    "business": {"size": 3, "quote": "pullquote", "bullet": "1"},
    "policy":   {"size": 3, "quote": "blockquote", "bullet": "a"},
    "tools":    {"size": 3, "quote": "expandable", "bullet": "1"},
}


def _theme(now: datetime) -> dict:
    return _THEMES[((now.timetuple().tm_yday * 4) + now.hour // 6) % len(_THEMES)]


# ── the bulletin ─────────────────────────────────────────────────────────
def build(stories: list[Story], summary_fa: str = "") -> dict:
    now = _tehran_now()
    clock = f"{now.hour:02d}:{now.minute:02d}".translate(_DIGITS)
    th = _theme(now)

    blocks: list[dict] = [
        anchor("top"),
        heading(f"{th['mark']} رادار هوش مصنوعی", size=1),
        para([italic(f"{_jalali(now)} — ساعت {clock} به وقت تهران"), "  •  ",
              code(f"{len(stories)}".translate(_DIGITS)), " خبر منتخب  •  ",
              {"type": "hashtag", "text": "#رادار_هوش_مصنوعی"}]),
    ]

    # Photos have to sit at the TOP LEVEL to be seen: art buried inside a
    # collapsed toggle is invisible until tapped, which is why the last bulletin
    # read as "هیچ عکسی نیست" even though it shipped two photos.
    with_art = [s for s in stories if s.image]
    lead = with_art[0] if with_art else None
    if lead is not None:
        blocks.append(photo(lead.image, caption={"text": [
            bold("تصویر شاخص  ·  "), lead.title_fa]}))

    if summary_fa:
        blocks.append(_digest_block(th["digest"], summary_fa))

    blocks.append(buttons([
        ("📡 کانال رادار", "https://t.me/ai_newsBY", th["accent"]),
        ("🔗 منبع خبر اول", stories[0].url if stories else "https://t.me/ai_newsBY", "link"),
    ]))

    # Headline board: titles, not a bare list. Each entry is its own heading with
    # a jump link into the full item, so the post is navigable from the top.
    blocks.append(divider())
    blocks.append(heading(f"{th['glance']} تیترهای این بولتن", size=2))
    for i, s in enumerate(stories, 1):
        blocks.append(heading([sup(f"{i}".translate(_DIGITS)), " ", s.title_fa], size=5))
        line: list = []
        if s.metric_label and s.metric_value:
            line += [marked(f" {s.metric_label}: {s.metric_value} "), "  "]
        line += [italic(s.source_fa), "  ·  ",
                 anchor_link("خواندن ↓", f"s{i}")]
        blocks.append(para(line))

    # Mid-message gallery: a second band of visible art, drawn from stories that
    # are NOT the lead so no picture is shown twice. `gallery` stays empty unless
    # the band is actually rendered — otherwise a single spare image would be
    # excluded from its own story toggle and vanish from the bulletin entirely.
    gallery: list[str] = []
    spare = [s.image for s in with_art[1:4]]
    if len(spare) >= 2:
        gallery = spare
        blocks.append(para(italic(th["rule"])))
        maker = collage if th["gallery"] == "collage" else slideshow
        blocks.append(maker(gallery, caption="قاب‌های امروز"))

    blocks.append(divider())

    # One toggle per section, one nested toggle per story.
    by_section: dict[str, list[Story]] = {}
    for s in stories:
        by_section.setdefault(s.section, []).append(s)

    rank = {id(s): i for i, s in enumerate(stories, 1)}
    first = True
    for key, label in config.SECTIONS:
        group = by_section.get(key) or []
        if not group:
            continue
        style = _SECTION_STYLE.get(key, _SECTION_STYLE["tools"])
        inner: list[dict] = []
        for s in group:
            i = rank[id(s)]
            inner.append(details(
                [code(f"{i}".translate(_DIGITS)), " ", bold(s.title_fa)],
                _story_blocks(s, i, style, with_image=s is not lead
                              and s.image not in gallery),
            ))
        count = f"{len(group)}".translate(_DIGITS)
        blocks.append(details([bold(label), "  ", italic(f"({count} خبر)")],
                              inner, is_open=first))
        first = False

    blocks.append(divider())
    blocks.append(details([bold("🗂 منابع این بولتن"), "  ",
                           italic("و پوشش دسته‌ها")], [
        table([["منبع", "خبر"]] + [[src, f"{n}".translate(_DIGITS)]
                                   for src, n in _source_counts(stories)],
              caption=italic("شماره خبرها بر پایه رتبه در همین بولتن")),
        para([bold("رصد این دوره: "), italic("چه دسته‌هایی خبر تازه داشتند")]),
        checklist([(label, bool(by_section.get(key)))
                   for key, label in config.SECTIONS]),
        para([italic("همه منابع سرچشمه اصلی‌اند؛ رادار بازنشر نمی‌کند."), "  ",
              anchor_link("بازگشت به بالا ↑", "top")]),
    ]))

    blocks.append(footer([
        "رادار هوش مصنوعی  •  ",
        link("@ai_newsBY", "https://t.me/ai_newsBY"),
        "  •  به‌روزرسانی هر ۶ ساعت",
    ]))

    # `skip_entity_detection` is what stops Telegram rewriting bare text into
    # links after delivery. Probed both ways on the live API: without it, a
    # paragraph reading "cs.AI و openai.com" comes back carrying two `url`
    # entities whose href IS the literal text (dead links — this is the bug that
    # shipped in post #98); with it, zero `url` entities are stored while an
    # explicit `url` element still works. Cheaper and cleaner than the earlier
    # U+2060 approach, which littered the text with invisible characters.
    return {"blocks": blocks, "is_rtl": True, "skip_entity_detection": True}


def _digest_block(kind: str, text: str, credit: str = "جمع‌بندی سردبیر") -> dict:
    if kind == "pullquote":
        return pullquote(text, credit=credit)
    if kind == "blockquote":
        return hard_quote([para(text)], credit=credit)
    return quote(text, credit=credit)


def _source_counts(stories: list[Story]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for s in stories:
        counts[f"{s.source_fa} ({s.source})"] = counts.get(f"{s.source_fa} ({s.source})", 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def _story_blocks(s: Story, rank: int, style: dict, with_image: bool = True) -> list[dict]:
    out: list[dict] = [anchor(f"s{rank}"), para(s.summary_fa)]

    if s.image and with_image:
        out.append(photo(s.image, caption={"text": [italic("تصویر: "), s.source_fa]}))

    # The headline number, given the weight of a heading rather than hidden in a
    # table cell.
    if s.metric_label and s.metric_value:
        out.append(heading([s.metric_label, ": ", marked(f" {s.metric_value} ")], size=4))

    if s.why_fa:
        out.append(_digest_block(style["quote"], s.why_fa, credit="چرا مهم است"))
    if s.impact_fa:
        out.append(pullquote(s.impact_fa, credit="اگر درست باشد"))

    if s.latex:
        out.append(para(italic("رابطه کلیدی:")))
        out.append(math(s.latex))

    if s.facts:
        out.append(details([bold("🔍 نکات کلیدی"), "  ",
                            italic(f"({len(s.facts)} نکته)".translate(_DIGITS))],
                           [numbered(s.facts, kind=style["bullet"])], is_open=True))

    out.append(table([["منبع", "اهمیت", "زمان انتشار"],
                      [s.source, f"{s.score:.0f}".translate(_DIGITS) + "/۱۰",
                       _tehran_time_of(s.published)]],
                     caption=italic("مشخصات خبر"), striped=False))

    if s.also_seen_in:
        out.append(para([italic("پوشش خبری دیگر: "), ", ".join(s.also_seen_in)]))
    # Buttons MUST carry a URL: probed live, a button with an empty url, with
    # `anchor_name`, or with no url at all is rejected with "Text buttons are not
    # allowed in the inline keyboard". Navigation therefore uses an anchor_link
    # entity in a paragraph, not a button.
    out.append(buttons([("خواندن متن کامل ↗", s.url, "primary"),
                        ("کانال رادار", "https://t.me/ai_newsBY", "link")]))
    out.append(para(anchor_link("بازگشت به تیترها ↑", "top")))
    return out


def _tehran_time_of(dt: datetime) -> str:
    local = dt.astimezone(timezone.utc) + timedelta(hours=3, minutes=30)
    return f"{local.hour:02d}:{local.minute:02d}".translate(_DIGITS)
