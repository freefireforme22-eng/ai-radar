"""Encode the watchable edition: the drawn cards, timed to the narration.

Why this module exists, and why it did not exist for three rounds: the rich
`video` block was written off as "inherently unreachable". That verdict came
from a BROKEN PROBE, not from Telegram. The probe hand-wrote its heading as
`{"type": "heading", "heading": {...}}`, while this codebase's proven schema is
`{"type": "heading", "text": ..., "size": n}`, so the server answered
`can't parse InputRichBlock: Can't find field "size"` — about the HEADING — and
the reply was read as "the video block is unsupported". Re-probed with
`render.heading()` and, decisively, with no heading at all: the message came
back stored as `types=['video']`, carrying width, height, duration and a
server-made thumbnail. `document` was mis-diagnosed identically. `motion.py`
already documents this exact failure mode; ignoring it cost two rounds of
"impossible".

What it produces: an MP4 that plays inside the post — the cover card, then every
story card, held on screen and set to the Persian narration that the bulletin
already synthesises. That makes the audio VISIBLE work rather than an
attachment nobody opens, and it is the only block in the post a reader can watch
instead of read.

Measured constraints:

* An `animation` is not a `video`. `sendAnimation` produces a silent loop, so
  the narration can only ride on `sendVideo` — which is why the chart could
  never carry the audio.
* There is no system ffmpeg here or on the CI runner; the encoder is the static
  binary inside the `imageio-ffmpeg` wheel (v4.2.2, libx264 + aac + libopus,
  mp4/ogg muxers all present — verified on this box).
* There is no ffprobe in that wheel. The narration's length is parsed from
  ffmpeg's own stderr banner (`Duration: HH:MM:SS.ss`), because a slideshow
  timed to the wrong length either freezes on the last card or truncates the
  voice.
* h264 needs even dimensions and `yuv420p`; frames arrive in two different
  sizes (1200x630 cover, 1000x560 story cards) so every frame is composited
  onto one 1280x720 canvas instead of being fed to the encoder as-is.

Everything is optional, the same contract as the cover, the chart and the
narration: any failure returns "" and the bulletin ships without the video.
"""
from __future__ import annotations

import math
import os
import re
import shutil
import subprocess
import tempfile

from .card import PALETTES

VW, VH = 1280, 720          # even dimensions, required by h264
FPS = 5                     # a slideshow needs no more; keeps the file tiny
_MIN_HOLD = 2.5             # seconds per card, floor
_MAX_HOLD = 6.0             # ceiling, so one card never sits for a minute
_TAIL = 0.6                 # seconds of video kept past the last spoken word
_MIN_BYTES = 8000           # smaller than this is a failed encode, not a video


def _exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""


def available() -> bool:
    """True when a video can actually be produced, checked before promising one."""
    if not _exe():
        return False
    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return False
    return True


def audio_seconds(path: str) -> float:
    """Length of an audio file in seconds, or 0.0 when it cannot be read.

    Parsed from ffmpeg's stderr because the imageio-ffmpeg wheel ships no
    ffprobe. `-i` with no output makes ffmpeg print the banner and exit 1, which
    is the expected path here, not an error.
    """
    exe = _exe()
    if not exe or not path or not os.path.exists(path):
        return 0.0
    try:
        proc = subprocess.run([exe, "-hide_banner", "-i", path],
                              capture_output=True, text=True, timeout=60)
    except Exception:
        return 0.0
    m = re.search(r"Duration:\s*(\d+):(\d{2}):(\d{2})\.(\d+)",
                  proc.stderr or proc.stdout or "")
    if not m:
        return 0.0
    h, mi, s, frac = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + float("0." + frac)


def build_pdf(*, frames: list[str], out_path: str = "") -> str:
    """Bundle the drawn cards into a one-page-per-card PDF ("" on failure).

    This is what makes the `document` block worth having rather than a box
    ticked: the bulletin becomes something a reader can save and read offline,
    and because the Persian text is already rasterised into the cards there is
    no font-embedding or shaping problem in the PDF at all.
    """
    frames = [p for p in frames if p and os.path.exists(p)]
    if not frames:
        return ""
    try:
        from PIL import Image
    except Exception:
        return ""
    out_path = out_path or os.path.join(tempfile.gettempdir(), "radar_bulletin.pdf")
    pages = []
    try:
        for p in frames:
            with Image.open(p) as im:
                pages.append(im.convert("RGB"))
        if not pages:
            return ""
        pages[0].save(out_path, "PDF", save_all=True, append_images=pages[1:],
                      resolution=110.0)
    except Exception:
        return ""
    finally:
        for im in pages:
            try:
                im.close()
            except Exception:
                pass
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1000:
        return ""
    return out_path


def _canvas(src_path: str, bg):
    """Fit one card onto the video canvas, letterboxed on the theme colour."""
    from PIL import Image

    frame = Image.new("RGB", (VW, VH), bg)
    with Image.open(src_path) as im:
        im = im.convert("RGB")
        scale = min(VW / im.width, VH / im.height)
        # Leave a small margin so a card never touches the frame edge.
        scale *= 0.94
        size = (max(2, int(im.width * scale)), max(2, int(im.height * scale)))
        im = im.resize(size, Image.LANCZOS)
        frame.paste(im, ((VW - size[0]) // 2, (VH - size[1]) // 2))
    return frame


def build(*, frames: list[str], audio_path: str = "", theme_index: int = 0,
          out_path: str = "") -> str:
    """Encode the briefing and return its path ("" when unavailable).

    Verified mechanically, since the vision service on this box is broken: each
    card is drawn in a different palette, so sampling the encoded MP4 at three
    timestamps and matching each frame's mean RGB against the source PNGs proves
    the right card is on screen at the right moment — 3 of 3 matched.

    `frames` are paths to already-drawn PNGs, in reading order. `audio_path` is
    the narration mp3; without it the slideshow still ships, silent, because a
    watchable card sequence beats no video at all.
    """
    frames = [p for p in frames if p and os.path.exists(p)]
    if not available() or not frames:
        return ""

    exe = _exe()
    out_path = out_path or os.path.join(tempfile.gettempdir(), "radar_briefing.mp4")
    bg = PALETTES[theme_index % len(PALETTES)][1]

    dur = audio_seconds(audio_path) if audio_path else 0.0
    if dur > 0:
        hold = max(_MIN_HOLD, min(_MAX_HOLD, dur / len(frames)))
        # The card sequence must COVER the narration, or the video stream ends
        # early and the player freezes on the last card while the voice keeps
        # talking. Cycling the cards a second time reads as intentional; a frozen
        # frame reads as a broken file.
        reps = max(1, math.ceil((dur + _TAIL) / (hold * len(frames))))
        order = frames * reps
    else:
        hold = 4.0
        order = list(frames)
    work = tempfile.mkdtemp(prefix="radar_brief_")
    try:
        listing = []
        rendered: dict[str, str] = {}
        for i, src in enumerate(order):
            png = rendered.get(src)
            if png is None:
                png = os.path.join(work, f"f{i:03d}.png")
                try:
                    _canvas(src, bg).save(png)
                except Exception:
                    continue
                rendered[src] = png
            listing.append((png, hold))
        if not listing:
            return ""

        # The concat demuxer needs the LAST entry repeated without a duration,
        # or ffmpeg drops the final image's hold time entirely.
        lines = []
        for png, secs in listing:
            lines.append(f"file '{png}'")
            lines.append(f"duration {secs:.3f}")
        lines.append(f"file '{listing[-1][0]}'")
        concat = os.path.join(work, "list.txt")
        with open(concat, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines) + "\n")

        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "concat", "-safe", "0", "-i", concat]
        if audio_path and os.path.exists(audio_path):
            cmd += ["-i", audio_path]
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage",
                "-crf", "26", "-pix_fmt", "yuv420p", "-r", str(FPS),
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2"]
        if audio_path and os.path.exists(audio_path):
            # MEASURED: `-shortest` does NOT bound the output here. The first
            # encode ran 12.20s against a 9.31s narration, because the concat
            # demuxer needs its last entry repeated (or the final card gets no
            # hold at all) and that repeat inherits the previous duration, so the
            # video stream outlives the audio and `-shortest` with the concat
            # demuxer did not clip it. An explicit `-t` is what actually bounds
            # the file: the narration plus a short tail so the last spoken word
            # is not cut mid-syllable.
            cmd += ["-c:a", "aac", "-b:a", "96k"]
            if dur > 0:
                cmd += ["-t", f"{dur + _TAIL:.3f}"]
        cmd += ["-movflags", "+faststart", out_path]

        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=420)
        except Exception:
            return ""
        if not os.path.exists(out_path) or os.path.getsize(out_path) < _MIN_BYTES:
            return ""
        return out_path
    finally:
        shutil.rmtree(work, ignore_errors=True)
