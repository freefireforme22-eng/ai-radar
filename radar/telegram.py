"""Telegram delivery + a plain-text fallback for non-premium clients."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from . import config


class TelegramError(RuntimeError):
    pass


def _api(method: str, payload: dict) -> dict:
    if not config.BOT_TOKEN:
        raise TelegramError("TG_BOT_TOKEN is not set")
    url = f"https://api.telegram.org/bot{config.BOT_TOKEN}/{method}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        return json.load(urllib.request.urlopen(req, timeout=60))
    except urllib.error.HTTPError as e:
        try:
            return json.load(e)
        except Exception:
            raise TelegramError(f"HTTP {e.code}") from e


def send_rich(rich_message: dict, chat_id: str | int | None = None) -> int:
    chat = chat_id or config.CHANNEL_ID
    res = _api("sendRichMessage", {"chat_id": chat, "rich_message": rich_message})
    if not res.get("ok"):
        raise TelegramError(res.get("description", json.dumps(res)[:200]))
    return res["result"]["message_id"]


def send_text(text: str, chat_id: str | int | None = None) -> int:
    chat = chat_id or config.CHANNEL_ID
    res = _api("sendMessage", {"chat_id": chat, "text": text,
                               "parse_mode": "HTML",
                               "link_preview_options": {"is_disabled": True}})
    if not res.get("ok"):
        raise TelegramError(res.get("description", json.dumps(res)[:200]))
    return res["result"]["message_id"]


def delete(message_id: int, chat_id: str | int | None = None) -> bool:
    chat = chat_id or config.CHANNEL_ID
    return bool(_api("deleteMessage", {"chat_id": chat,
                                       "message_id": message_id}).get("ok"))


def _upload(path: str, field: str, filename: str, mime: str,
            chat_id: str | int | None = None, **fields) -> dict:
    """POST a local file through a normal send method and return `result`.

    Rich messages have no upload endpoint: the only way to get a `file_id` is to
    send the file with a classic method first. Shared by audio and photo so the
    multipart assembly exists once.
    """
    if not config.BOT_TOKEN:
        raise TelegramError("TG_BOT_TOKEN is not set")
    chat = chat_id or config.CHANNEL_ID
    boundary = "----radarboundary7d1f"
    form = {"chat_id": str(chat), "disable_notification": "true"}
    form.update({k: str(v) for k, v in fields.items() if v})

    body = b""
    for key, value in form.items():
        body += (f"--{boundary}\r\nContent-Disposition: form-data; "
                 f"name=\"{key}\"\r\n\r\n{value}\r\n").encode("utf-8")
    with open(path, "rb") as fh:
        blob = fh.read()
    body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; "
             f"filename=\"{filename}\"\r\n"
             f"Content-Type: {mime}\r\n\r\n").encode("utf-8")
    body += blob + f"\r\n--{boundary}--\r\n".encode("utf-8")

    # The send method follows the FIELD name — a lookup, not an if/else, because
    # the two-way version silently sent a GIF to `sendPhoto` ("there is no photo
    # in the request") the moment a third media kind was added.
    method = {"audio": "sendAudio",
              "animation": "sendAnimation",
              "photo": "sendPhoto"}[field]
    url = f"https://api.telegram.org/bot{config.BOT_TOKEN}/{method}"
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    try:
        res = json.load(urllib.request.urlopen(req, timeout=120))
    except urllib.error.HTTPError as e:
        try:
            res = json.load(e)
        except Exception:
            raise TelegramError(f"HTTP {e.code}") from e
    if not res.get("ok"):
        raise TelegramError(res.get("description", f"{field} upload failed"))
    return res["result"]


def upload_photo(path: str, chat_id: str | int | None = None) -> str:
    """Upload a local PNG and return its `file_id`, deleting the carrier.

    Probed live: a rich `photo` block accepts a `file_id` in `media`, exactly as
    `audio` does — the stored message came back with block types
    `['heading', 'photo']`. So the drawn cover needs no public hosting.

    `sendPhoto` returns an array of thumbnail sizes; the LAST entry is the
    largest, and that is the id to reuse. Picking sizes[0] would ship a
    ~90px-wide cover.
    """
    result = _upload(path, "photo", "cover.png", "image/png", chat_id)
    sizes = result.get("photo") or []
    file_id = sizes[-1]["file_id"] if sizes else ""
    try:
        delete(result["message_id"], chat_id or config.CHANNEL_ID)
    except Exception:
        pass                  # the id is what matters; a stray carrier is not fatal
    return file_id


def upload_audio(path: str, chat_id: str | int | None = None,
                 *, title: str = "") -> str:
    """Upload an mp3 and return its `file_id`, deleting the carrier message.

    Rich-message media blocks take an `InputMedia` object, and `media` accepts a
    `file_id` — probed live — so audio does not need public hosting. But there
    is no upload endpoint for rich messages: the only way to mint a `file_id` is
    to send the file through a normal method first. So this posts it silently,
    grabs the id, and deletes the carrier. The `file_id` stays valid afterwards.

    Sent with `sendAudio`, not `sendVoice`: `caption` survives on an `audio`
    rich block and is dropped on `voice_note` (measured), and the audio block
    keeps the title/performer metadata a bulletin wants.
    """
    result = _upload(path, "audio", "narration.mp3", "audio/mpeg",
                     chat_id, title=title)
    file_id = (result.get("audio") or {}).get("file_id", "")
    try:
        delete(result["message_id"], chat_id or config.CHANNEL_ID)
    except Exception:
        pass          # the id is what matters; a stray carrier is not fatal
    return file_id


def upload_animation(path: str, chat_id: str | int | None = None) -> str:
    """Upload the loop and return its `animation` file_id.

    MP4 in, MP4 out at full resolution. A GIF would also be accepted and
    transcoded, but Telegram downscales GIFs to 320px wide (measured), so
    `motion.py` encodes h264 first and this only forwards the file. Same
    carrier-and-delete trick as the audio and cover uploads, because rich
    messages have no upload endpoint of their own.
    """
    gif = path.endswith(".gif")
    result = _upload(path, "animation",
                     "chart.gif" if gif else "chart.mp4",
                     "image/gif" if gif else "video/mp4", chat_id)
    file_id = (result.get("animation") or {}).get("file_id", "")
    try:
        delete(result["message_id"], chat_id or config.CHANNEL_ID)
    except Exception:
        pass          # the id is what matters; a stray carrier is not fatal
    return file_id


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def plain_fallback(stories, summary_fa: str = "") -> str:
    """HTML rendering used when sendRichMessage is unavailable.

    Rich messages only render fully for Premium users; this keeps the channel
    readable everywhere and is also the emergency path if the rich payload is
    rejected. It mirrors the rich renderer's information — including the
    analytical fields — so the fallback is not a downgrade in substance.
    """
    lines = ["<b>🛰 رادار هوش مصنوعی</b>", ""]
    if summary_fa:
        lines += [f"<blockquote>{_esc(summary_fa)}</blockquote>", ""]
    by_section: dict[str, list] = {}
    for s in stories:
        by_section.setdefault(s.section, []).append(s)
    for key, label in config.SECTIONS:
        group = by_section.get(key) or []
        if not group:
            continue
        lines.append(f"<b>{_esc(label)}</b>")
        for s in group:
            lines.append(f"• <a href=\"{_esc(s.url)}\"><b>{_esc(s.title_fa)}</b></a>")
            lines.append(_esc(s.summary_fa))
            metric = getattr(s, "metric_label", ""), getattr(s, "metric_value", "")
            if metric[0] and metric[1]:
                lines.append(f"   <b>{_esc(metric[0])}:</b> <code>{_esc(metric[1])}</code>")
            if getattr(s, "impact_fa", ""):
                lines.append(f"   <i>اگر درست باشد: {_esc(s.impact_fa)}</i>")
            for n, f in enumerate(s.facts, 1):
                lines.append(f"   {n}. {_esc(f)}")
            lines.append(f"   <i>{_esc(s.source)}</i>")
            lines.append("")
    lines.append("<i>@ai_newsBY — هر ۶ ساعت</i>")
    return "\n".join(lines)[:4000]
