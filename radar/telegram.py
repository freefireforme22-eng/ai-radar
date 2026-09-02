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


def _esc(text: str) -> str:
    return (text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def plain_fallback(stories, summary_fa: str = "") -> str:
    """HTML rendering used when sendRichMessage is unavailable.

    Rich messages only render fully for Premium users; this keeps the channel
    readable everywhere and is also the emergency path if the rich payload is
    rejected.
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
            if s.facts:
                lines += [f"   ☑️ {_esc(f)}" for f in s.facts]
            lines.append(f"   <i>{_esc(s.source)}</i>")
            lines.append("")
    lines.append("<i>@ai_newsBY — هر ۶ ساعت</i>")
    return "\n".join(lines)[:4000]
