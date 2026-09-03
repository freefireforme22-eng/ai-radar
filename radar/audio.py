"""Persian audio narration for the bulletin.

Why this exists: the standing complaint is «خیلی خشک و خالیه فقط متنه» — dry
text only. Pictures fixed part of it; a narrated version fixes the rest, because
it is the one block type that turns the bulletin into something you can consume
without reading.

How it works
------------
1. `edge-tts` synthesises Persian speech offline-ish (Microsoft's public
   endpoint, no API key, no account).
2. The mp3 is uploaded once with `sendAudio` to obtain a Telegram `file_id`,
   and that throwaway message is deleted immediately.
3. The `file_id` goes into an `audio` rich block.

Measured facts behind these choices (probed live, see also render.py):
  * `{"type": "audio", "audio": {"type": "audio", "media": "<file_id>"}}` is
    accepted and stored — a `file_id` works, so no public hosting is needed.
  * `caption` survives on an `audio` block (it does NOT survive on
    `voice_note`), so the narration can be labelled.
  * Only two Persian voices exist: `fa-IR-DilaraNeural` (female) and
    `fa-IR-FaridNeural` (male).
  * Peak RSS of a synthesis run: 38MB. That matters — the box is capped at
    953.7MB by the cgroup, so this must not be a memory event.

Everything here is best-effort: any failure returns "" and the bulletin ships
silently without audio. A news channel must never go quiet because a TTS
endpoint had a bad minute.
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile

from . import telegram

# Rotating narrator, so consecutive bulletins do not sound identical either.
VOICES = ("fa-IR-DilaraNeural", "fa-IR-FaridNeural")

_MAX_CHARS = 1800          # ~2 minutes of speech; keeps the upload small
_MIN_BYTES = 2000          # anything smaller is a failed synthesis, not audio


def to_voice(mp3_path: str) -> str:
    """Transcode an mp3 to OGG/Opus for `sendVoice`. "" when unavailable.

    A voice note is a different reading experience from an audio file — it plays
    inline as a bubble with a waveform instead of sitting in a player row — and
    `sendVoice` accepts nothing but Opus in an OGG container (a renamed mp3 is
    rejected). There is no system ffmpeg on this box or on the CI runner, so the
    static binary shipped inside the `imageio_ffmpeg` wheel is used, exactly as
    `motion.py` does for the animated chart.
    """
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return ""
    out = os.path.join(tempfile.gettempdir(), "radar_narration.ogg")
    try:
        proc = subprocess.run(
            [exe, "-y", "-hide_banner", "-loglevel", "error", "-i", mp3_path,
             "-c:a", "libopus", "-b:a", "32k", "-ar", "48000", "-ac", "1", out],
            capture_output=True, timeout=180)
    except Exception:
        return ""
    if proc.returncode != 0 or not os.path.exists(out) \
            or os.path.getsize(out) < _MIN_BYTES:
        return ""
    return out


def narration_text(summary_fa: str, headlines: list[str], clock: str = "") -> str:
    """Build the script: the editor's digest, then the headlines, spoken.

    Digits are read aloud badly by TTS when they are Persian numerals mixed with
    Latin ones, so the script deliberately carries no numbering — the reader
    hears "و بعد" style flow instead of "۱. ۲. ۳.".
    """
    parts = ["رادار هوش مصنوعی."]
    if clock:
        parts.append(f"گزارش ساعت {clock} به وقت تهران.")
    if summary_fa:
        parts.append(summary_fa)
    if headlines:
        parts.append("تیترهای این دوره:")
        parts.extend(f"{h}." for h in headlines[:5])
    parts.append("متن کامل خبرها در همین پیام.")
    script = " ".join(p.strip() for p in parts if p and p.strip())
    return script[:_MAX_CHARS]


def synthesise(script: str, voice: str = VOICES[0]) -> str:
    """Render `script` to an mp3 on disk. Returns the path, or "" on failure."""
    if not script.strip():
        return ""
    try:
        import edge_tts
    except ImportError:
        return ""

    path = os.path.join(tempfile.gettempdir(), "radar_narration.mp3")

    async def _run() -> None:
        await edge_tts.Communicate(script, voice).save(path)

    try:
        asyncio.run(_run())
    except Exception:
        return ""
    if not os.path.exists(path) or os.path.getsize(path) < _MIN_BYTES:
        return ""
    return path


def narrate(summary_fa: str, headlines: list[str], chat_id, *,
            clock: str = "", voice: str = VOICES[0], title: str = "رادار هوش مصنوعی",
            as_voice_note: bool = False) -> tuple[str, str]:
    """Full path: script -> mp3 -> uploaded -> `file_id`.

    Returns `(file_id, kind)` where kind is "voice_note" or "audio"; `("", "")`
    if anything fails. A voice note is preferred when asked for because it reads
    as a spoken message rather than an attachment, but the transcode needs
    ffmpeg — when that is missing the mp3 still ships as an `audio` block instead
    of the bulletin losing its narration.
    """
    path = synthesise(narration_text(summary_fa, headlines, clock), voice)
    if not path:
        return "", ""
    ogg = to_voice(path) if as_voice_note else ""
    try:
        if ogg:
            file_id = telegram.upload_voice(ogg, chat_id)
            if file_id:
                return file_id, "voice_note"
        return telegram.upload_audio(path, chat_id, title=title), "audio"
    except Exception:
        return "", ""
    finally:
        for p in (path, ogg):
            if not p:
                continue
            try:
                os.remove(p)
            except OSError:
                pass
