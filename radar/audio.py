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
import tempfile

from . import telegram

# Rotating narrator, so consecutive bulletins do not sound identical either.
VOICES = ("fa-IR-DilaraNeural", "fa-IR-FaridNeural")

_MAX_CHARS = 1800          # ~2 minutes of speech; keeps the upload small
_MIN_BYTES = 2000          # anything smaller is a failed synthesis, not audio


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
            clock: str = "", voice: str = VOICES[0], title: str = "رادار هوش مصنوعی") -> str:
    """Full path: script -> mp3 -> uploaded -> `file_id`. "" if anything fails."""
    path = synthesise(narration_text(summary_fa, headlines, clock), voice)
    if not path:
        return ""
    try:
        return telegram.upload_audio(path, chat_id, title=title)
    except Exception:
        return ""
    finally:
        try:
            os.remove(path)
        except OSError:
            pass
