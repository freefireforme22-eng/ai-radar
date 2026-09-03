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

1. **PIL on this box has no complex-script shaping** — `PIL.features` reports
   raqm=False, harfbuzz=False, fribidi=False. Arabic script drawn straight from
   a Python string comes out as isolated, left-to-right letters. So text is
   pre-shaped: `arabic_reshaper` maps letters to their presentation forms and
   `python-bidi` reorders visually. Verified mechanically on «میلیارد» — shaped
   gives 172px of ink with 1 interior gap, unshaped gives 240px with 5 gaps;
   joined cursive is exactly what the narrower, gapless run means.

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


def _shape(text: str) -> str:
    """Persian string → visually ordered presentation forms.

    Without this every card would read as disconnected mirror-image letters, and
    it would still *look* like text at thumbnail size — which is exactly the kind
    of defect that shipped to the channel before.
    """
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
