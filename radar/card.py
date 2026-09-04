"""Generate the bulletin's cover card — the one place we control COLOUR.

The user's request was explicit: «اگر هر پیام رنگ و ویژگی خاص خودشو داشته باشه
بهتر میشه». Rich messages expose no colour or font control at all: the only
`accent` field lives on buttons, headings have a size and nothing else. Probed
across all 24 documented block types — none carries a colour.

So colour has to arrive as pixels. This module draws a cover image locally, per
theme, with its own palette and its own geometric motif, uploads it once to mint
a `file_id` (measured: a rich `photo` block accepts a `file_id` in `media`, so no
public hosting is needed), and the bulletin leads with it. Six themes, six
palettes: a reader scrolling the channel sees a different colour every slot.

Two measured constraints shape the implementation:

1. **Shaping must match the layout engine, and the engine CHANGED under us.**
   The original measurement (Pillow without raqm on this box) is now stale:
   Pillow 12.3.0 — here and on the CI runner — ships the raqm layout engine
   (`ImageFont.truetype(...).layout_engine == 1`, `features.check("raqm")` is
   True). raqm performs bidi reordering AND Arabic shaping itself, so feeding
   it pre-shaped text (arabic_reshaper + python-bidi) double-transforms it:
   the letters stay joined but the word order comes out MIRRORED. Measured on
   the live channel: post 185's video frame OCRs as «ی‌عونصم ش‌وه رادار» —
   exactly the «به هم ریخته و اصلا خونده نمیشه» the user reported. Control:
   the same string drawn raw through raqm OCRs correctly. Therefore `_shape`
   is now ADAPTIVE — an identity function when raqm is present, the legacy
   reshaper pipeline only when it is not. Every draw site already routes
   through `_shape`, so one switch fixes all of them.

2. **Vazirmatn carries no emoji or geometric glyphs** — measured ink for
   U+1F4CA, U+26A0, U+25A0, U+25CF is all zero. The theme marks used in the text
   bulletin cannot be drawn into the card, so each theme's motif is drawn with
   PIL primitives instead (arcs, grids, crosses).

Everything here is optional. Any failure — missing Pillow, missing font, missing
shaper — returns an empty path and the bulletin ships exactly as before, the same
contract the narration uses.
"""
from __future__ import annotations

import os
import tempfile

# Palettes, one per theme index, matching the marks in render._THEMES:
# 🛰 🌐 🧭 🔭 📡 🛠. Each is (top, bottom, accent, ink) — a vertical gradient plus
# the accent used for rules and the leading bars.
PALETTES = [
    ((14, 26, 51), (7, 12, 26), (94, 174, 255), (236, 243, 255)),    # 🛰 deep space blue
    ((9, 38, 32), (5, 16, 14), (74, 222, 160), (232, 252, 244)),     # 🌐 signal green
    ((46, 14, 22), (20, 7, 11), (255, 122, 122), (255, 236, 236)),   # 🧭 alert crimson
    ((25, 17, 46), (11, 8, 22), (176, 140, 255), (240, 235, 255)),   # 🔭 violet observatory
    ((10, 32, 46), (5, 15, 22), (86, 205, 230), (230, 249, 255)),    # 📡 cyan array
    ((44, 30, 8), (20, 14, 4), (255, 186, 80), (255, 245, 227)),     # 🛠 workshop amber
]

_FONT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
_BOLD = os.path.join(_FONT_DIR, "Vazirmatn-Bold.ttf")
_REG = os.path.join(_FONT_DIR, "Vazirmatn-Regular.ttf")

W, H = 1200, 630


def _raqm_active() -> bool:
    """True when Pillow will shape+bidi the text itself at draw time.

    Pillow wheels >= 10.3 bundle libraqm. When the raqm layout engine is active,
    `draw.text()` already applies bidi reordering and Arabic joining — feeding
    it arabic_reshaper/python-bidi output transforms the string TWICE and the
    render comes out mirrored (letters joined, word order reversed). Measured
    live: the post-185 video frame — drawn through the old always-reshape path
    on raqm-Pillow — OCRs as «ی‌عونصم ش‌وه رادار»; the same string drawn raw
    through raqm OCRs correctly. The docstring history that claimed raqm=False
    described an older container, not the current runner.
    """
    try:
        from PIL import features
        return bool(features.check("raqm"))
    except Exception:
        return False


def _shape(text: str) -> str:
    """Persian string → glyphs ready for `draw.text`, ADAPTIVELY.

    With raqm (current venv AND CI): return the text untouched — raqm does the
    bidi reordering and contextual joining, and pre-shaped input is what
    produced the mirrored «به هم ریخته» output the user reported. Without raqm
    (the original environment this module was written for): pre-shape through
    arabic_reshaper + python-bidi, because raw Arabic-script text drawn via the
    BASIC engine comes out as disconnected left-to-right letters.
    """
    if _raqm_active():
        return text
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(text))


def available() -> bool:
    """True when a card can actually be drawn, checked before promising one."""
    try:
        import arabic_reshaper  # noqa: F401
        from bidi.algorithm import get_display  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return os.path.exists(_BOLD) and os.path.exists(_REG)


def _gradient(draw, top, bottom):
    for y in range(H):
        t = y / (H - 1)
        draw.line([(0, y), (W, y)],
                  fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))


def _motif(draw, idx, accent):
    """A per-theme geometric signature, since the font has no emoji outlines.

    Drawn faint and bled off the left edge so it reads as texture behind the
    text rather than a competing object.
    """
    a = accent + (60,) if len(accent) == 3 else accent
    if idx == 0:                                    # 🛰 radar sweeps
        for r in range(120, 620, 90):
            draw.arc([-r, H // 2 - r, r, H // 2 + r], 300, 60, fill=a, width=3)
    elif idx == 1:                                  # 🌐 mesh
        for x in range(0, 460, 58):
            draw.line([(x, 0), (x, H)], fill=a, width=1)
        for y in range(0, H, 58):
            draw.line([(0, y), (460, y)], fill=a, width=1)
    elif idx == 2:                                  # 🧭 compass rose
        cx, cy = 210, H // 2
        for rad in (70, 150, 230):
            draw.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=a, width=2)
        draw.line([(cx - 250, cy), (cx + 250, cy)], fill=a, width=2)
        draw.line([(cx, cy - 250), (cx, cy + 250)], fill=a, width=2)
    elif idx == 3:                                  # 🔭 star field + lens
        import random
        rng = random.Random(7)
        for _ in range(90):
            x, y = rng.randrange(0, 520), rng.randrange(0, H)
            s = rng.choice((1, 1, 2))
            draw.ellipse([x, y, x + s, y + s], fill=a)
        draw.ellipse([60, H // 2 - 170, 400, H // 2 + 170], outline=a, width=3)
    elif idx == 4:                                  # 📡 dish rings
        for r in range(60, 520, 70):
            draw.arc([160 - r, H // 2 - r, 160 + r, H // 2 + r], 250, 110, fill=a, width=2)
    else:                                           # 🛠 diagonal hatch
        for x in range(-H, 520, 44):
            draw.line([(x, 0), (x + H, H)], fill=a, width=1)


SW, SH = 1000, 560          # per-story card: taller ratio than the wide cover


def _wrap(draw, text: str, font, max_w: int, max_lines: int) -> list[str]:
    """Greedy word wrap measured on the SHAPED string.

    Wrapping must happen on the raw words but be measured after reshaping, or
    the widths are wrong for every joined Persian form and lines overflow the
    card. Returns already-shaped lines, ready to draw.
    """
    words = text.split()
    lines: list[str] = []
    cur: list[str] = []
    for word in words:
        trial = " ".join(cur + [word])
        if cur and draw.textlength(_shape(trial), font=font) > max_w:
            lines.append(" ".join(cur))
            cur = [word]
            if len(lines) == max_lines:
                break
        else:
            cur.append(word)
    if cur and len(lines) < max_lines:
        lines.append(" ".join(cur))
    if not lines:
        return []
    # Mark truncation on the last line when words were left over.
    used = sum(len(l.split()) for l in lines)
    if used < len(words):
        lines[-1] = lines[-1] + " …"
    return [_shape(l) for l in lines]


def build_story(*, rank: int, rank_fa: str, section_fa: str, title_fa: str,
                source_fa: str, metric: str = "", palette: int = 0,
                out_path: str = "") -> str:
    """Draw a card for ONE story and return its path ("" when unavailable).

    Why this exists: auditing the live posts per CARD rather than per post showed
    the bulletin was still mostly text where it is actually read. Post 141 had
    four story cards and NONE carried a picture; post 152, six cards and one.
    The post-level photo count looked healthy only because the cover, the
    gallery band and the lead image all sit at the top — the cards a reader
    opens were bare. Feeds are the cause: arXiv and most research sources ship
    no art at all, so `Story.image` is empty for them and no amount of feed
    tuning fixes it.

    The palette is chosen by the story's RANK, not by the bulletin theme, so
    cards inside one post differ from each other in colour — the complaint
    «قالب پیامای الان همه شون شبیه همه» applies inside a post as much as
    between posts.
    """
    if not available():
        return ""
    try:
        from PIL import Image, ImageDraw, ImageFont

        top, bottom, accent, ink = PALETTES[palette % len(PALETTES)]
        img = Image.new("RGB", (SW, SH), top)
        draw = ImageDraw.Draw(img, "RGBA")

        for y in range(SH):
            t = y / (SH - 1)
            draw.line([(0, y), (SW, y)],
                      fill=tuple(round(a + (b - a) * t) for a, b in zip(top, bottom)))

        faint = accent + (46,)
        # A different geometric signature per card, keyed off the rank so two
        # neighbouring cards never carry the same texture.
        which = rank % 4
        if which == 0:
            for r in range(70, 720, 78):
                draw.arc([-r, SH // 2 - r, r, SH // 2 + r], 270, 90,
                         fill=faint, width=3)
        elif which == 1:
            for x in range(-SH, 560, 52):
                draw.line([(x, 0), (x + SH, SH)], fill=faint, width=2)
        elif which == 2:
            for i in range(9):
                s = 40 + i * 52
                draw.rectangle([30, SH - s - 30, 30 + s, SH - 30],
                               outline=faint, width=2)
        else:
            for gy in range(0, SH, 46):
                for gx in range(0, 520, 46):
                    draw.ellipse([gx, gy, gx + 5, gy + 5], fill=faint)

        f_rank = ImageFont.truetype(_BOLD, 190)
        f_sec = ImageFont.truetype(_REG, 32)
        f_title = ImageFont.truetype(_BOLD, 56)
        f_foot = ImageFont.truetype(_REG, 30)

        margin = 64

        def right(text_shaped: str, font, y, fill):
            w = draw.textlength(text_shaped, font=font)
            draw.text((SW - margin - w, y), text_shaped, font=font, fill=fill)

        # The rank, oversized and bled off the left edge: an instant visual
        # anchor that also tells the reader where they are in the bulletin.
        draw.text((margin - 18, SH - 258), rank_fa, font=f_rank, fill=faint)

        right(_shape(section_fa), f_sec, 52, accent)
        draw.rectangle([SW - margin - 190, 104, SW - margin, 110], fill=accent)

        y = 148
        for line in _wrap(draw, title_fa, f_title, SW - 2 * margin - 40, 4):
            right(line, f_title, y, ink)
            y += 74

        if metric:
            badge = _shape(metric)
            bw = draw.textlength(badge, font=f_sec)
            bx, by = SW - margin - bw - 26, min(y + 14, SH - 132)
            draw.rounded_rectangle([bx, by, bx + bw + 26, by + 52], 14,
                                   fill=accent)
            draw.text((bx + 13, by + 8), badge, font=f_sec, fill=bottom)

        right(_shape(f"رادار هوش مصنوعی  ·  {source_fa}"), f_foot,
              SH - margin - 12, accent)

        path = out_path or os.path.join(tempfile.gettempdir(),
                                        f"radar_story_{rank}.png")
        img.save(path, "PNG", optimize=True)
        return path
    except Exception:
        return ""


def build(*, theme_index: int, date_fa: str, clock: str, count_fa: str,
          headlines: list[str], out_path: str = "") -> str:
    """Draw the cover and return its path (empty string when unavailable)."""
    if not available():
        return ""
    try:
        from PIL import Image, ImageDraw, ImageFont

        top, bottom, accent, ink = PALETTES[theme_index % len(PALETTES)]
        img = Image.new("RGB", (W, H), top)
        draw = ImageDraw.Draw(img, "RGBA")
        _gradient(draw, top, bottom)
        _motif(draw, theme_index % len(PALETTES), accent)

        f_title = ImageFont.truetype(_BOLD, 78)
        f_meta = ImageFont.truetype(_REG, 34)
        f_head = ImageFont.truetype(_REG, 40)

        margin = 70

        def rtl(text, font, y, fill, *, bar=False):
            """Right-align, because a left-aligned Persian line looks broken."""
            s = _shape(text)
            w = draw.textlength(s, font=font)
            x = W - margin - w
            if bar:
                draw.rectangle([W - margin + 12, y + 8, W - margin + 20, y + 44],
                               fill=accent)
            draw.text((x, y), s, font=font, fill=fill)
            return w

        rtl("رادار هوش مصنوعی", f_title, 78, ink)
        draw.rectangle([W - margin - 300, 186, W - margin, 192], fill=accent)
        rtl(f"{date_fa} — ساعت {clock} به وقت تهران  ·  {count_fa} خبر منتخب",
            f_meta, 214, ink)

        y = 306
        for line in headlines[:3]:
            text = line if len(line) <= 46 else line[:45].rstrip() + "…"
            rtl(text, f_head, y, ink, bar=True)
            y += 74

        path = out_path or os.path.join(tempfile.gettempdir(), "radar_cover.png")
        img.save(path, "PNG", optimize=True)
        return path
    except Exception:
        return ""            # a cover is a bonus; never block the bulletin
