"""Draw the bulletin's animated chart — the only MOVING element in the post.

Why this exists: «خیلی خشک و خالیه فقط متنه». A cover card gave each post its own
colour, but the whole bulletin was still motionless. Rich messages do support an
`animation` block, and it autoplays (the server echoes `need_autoplay: true`),
so a short loop is the one way to put motion in a channel post.

That support was nearly missed. An earlier probe sent
`{"type":"animation", ...}` after a hand-written heading and came back with
`Bad Request: can't parse InputRichBlock: Can't find field "size"`, which read as
"the animation block needs an undocumented size". It did not: `size` is a
REQUIRED field of InputRichBlockSectionHeading, and the failing block was the
HEADING. Telegram names the missing field but never says which block it belongs
to, so a malformed block earlier in the array makes a later, perfectly valid
block look unsupported. Re-probed with `heading(size=2)`: animation, collage,
document, table cell align/colspan and button align are all accepted.

The animation charts something real — how the slot's stories split across
sections — rather than being decorative motion, and it is drawn in the same
palette as the cover so a post stays visually one piece.

Constraints measured on this box (identical to card.py): PIL has no complex-script
shaping (raqm/harfbuzz/fribidi all False), so Persian is pre-shaped through
arabic_reshaper + python-bidi; and Vazirmatn has no emoji outlines, so bars and
ticks are PIL primitives.

**The loop must be uploaded as MP4, not GIF.** `sendAnimation` accepts a GIF and
does transcode it to `video/mp4`, but it also downscales it: a 720x420 GIF came
back stored as **320x188**, and passing explicit `width`/`height` form fields
changed nothing (probed both ways). Persian labels at 320px wide are unreadable,
which defeats the point of charting anything. The same frames encoded to h264
first come back stored at the full **720x420**. There is no system ffmpeg here or
on the CI runner, so the encoder is the static binary shipped inside the
`imageio-ffmpeg` wheel (`imageio_ffmpeg.get_ffmpeg_exe()`, v4.2.2, libx264
present). If that import is missing the module degrades to GIF rather than
failing — a small chart beats no chart.

Everything is optional: any failure returns "" and the bulletin ships without
motion, the same contract as the narration and the cover.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from .card import PALETTES, _BOLD, _REG, _shape, available  # same deps, same guard

W, H = 720, 420
FRAMES = 16
FPS = 12
# The final frame is the only one carrying the numbers, so it is held. In an MP4
# that means repeating it (constant framerate); the GIF fallback uses a per-frame
# duration instead, because Pillow DROPS trailing byte-identical frames
# (measured: 22 frames in, 16 out).
HOLD_MS = 900


def _bar_chart(draw, *, t: float, rows: list[tuple[str, int]], palette, fonts) -> None:
    """One frame: horizontal bars filling right-to-left to fraction `t`."""
    top, bottom, accent, ink = palette
    f_title, f_label = fonts
    margin = 46
    axis_x = W - margin - 150          # bars start (right) after the label column

    title = _shape("پخش خبرهای این بولتن")
    draw.text((W - margin - draw.textlength(title, font=f_title), 26),
              title, font=f_title, fill=ink)
    draw.line([(margin, 92), (W - margin, 92)], fill=accent + (110,), width=2)

    biggest = max((n for _, n in rows), default=1) or 1
    y = 118
    span = axis_x - margin - 40
    for label, n in rows:
        shaped = _shape(label)
        draw.text((W - margin - draw.textlength(shaped, font=f_label), y + 4),
                  shaped, font=f_label, fill=ink)
        full = span * (n / biggest)
        w = full * t
        if w >= 1:
            draw.rectangle([axis_x - w, y, axis_x, y + 34], fill=accent)
        # the count rides just past the bar's leading (left) edge
        if t > 0.75:
            cnt = _shape(str(n).translate(str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")))
            draw.text((axis_x - full - 14 - draw.textlength(cnt, font=f_label), y + 4),
                      cnt, font=f_label, fill=accent)
        y += 62


def _encode(frames, out_path: str) -> str:
    """Frames -> h264 MP4 via the bundled static ffmpeg; GIF if it is missing.

    Telegram downscales uploaded GIFs to 320px wide, so the MP4 path is the one
    that keeps the Persian labels legible. The GIF branch stays as a degradation
    step, never as the preferred output.
    """
    from PIL import Image  # noqa: F401  (frames are already PIL images)

    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = ""

    if not exe:
        gif = os.path.splitext(out_path)[0] + ".gif"
        frames[0].save(gif, save_all=True,
                       append_images=[f.convert("P", palette=1, colors=64)
                                      for f in frames[1:]],
                       duration=[80] * (len(frames) - 1) + [HOLD_MS],
                       loop=0, optimize=True)
        return gif

    work = tempfile.mkdtemp(prefix="radar_motion_")
    try:
        # The last frame carries the numbers, so it is repeated to hold on screen;
        # with a constant framerate that is the only way to pause an MP4 loop.
        hold = max(1, round(HOLD_MS / (1000 / FPS)))
        seq = list(frames) + [frames[-1]] * hold
        for i, frame in enumerate(seq):
            frame.save(os.path.join(work, f"f{i:03d}.png"))
        cmd = [exe, "-y", "-loglevel", "error", "-framerate", str(FPS),
               "-i", os.path.join(work, "f%03d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               # h264 needs even dimensions; W/H are even but a future size may not be
               "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", out_path]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)


def build(*, theme_index: int, rows: list[tuple[str, int]], out_path: str = "") -> str:
    """Render the loop and return its path (empty string when unavailable)."""
    if not available() or not rows:
        return ""
    try:
        from PIL import Image, ImageDraw, ImageFont

        palette = PALETTES[theme_index % len(PALETTES)]
        top, bottom, accent, ink = palette
        fonts = (ImageFont.truetype(_BOLD, 40), ImageFont.truetype(_REG, 30))

        frames = []
        for i in range(FRAMES):
            # ease-out: fast at the start, settling at the end, so the eye reads
            # the final proportions rather than a constant-speed wipe
            t = 1 - (1 - (i + 1) / FRAMES) ** 2
            img = Image.new("RGB", (W, H), top)
            d = ImageDraw.Draw(img, "RGBA")
            for y in range(H):
                k = y / (H - 1)
                d.line([(0, y), (W, y)],
                       fill=tuple(round(a + (b - a) * k) for a, b in zip(top, bottom)))
            _bar_chart(d, t=t, rows=rows, palette=palette, fonts=fonts)
            frames.append(img)

        path = out_path or os.path.join(tempfile.gettempdir(), "radar_motion.mp4")
        return _encode(frames, path)
    except Exception:
        return ""
