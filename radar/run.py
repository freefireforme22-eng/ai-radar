"""Entry point: collect → triage → localise → render → publish.

Usage:
    python -m radar.run                 # publish a bulletin
    python -m radar.run --dry-run       # build it, print stats, post nothing
    python -m radar.run --preview CHAT  # post to a private chat first
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone

from . import audio, card, config, enrich, render, sources, telegram


def load_seen() -> dict:
    try:
        with open(config.STATE_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_seen(seen: dict) -> None:
    os.makedirs(os.path.dirname(config.STATE_PATH) or ".", exist_ok=True)
    if len(seen) > config.STATE_KEEP:
        keep = sorted(seen.items(), key=lambda kv: kv[1], reverse=True)[:config.STATE_KEEP]
        seen = dict(keep)
    with open(config.STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(seen, f)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="AI Radar — Persian AI news bulletin")
    ap.add_argument("--dry-run", action="store_true", help="build but do not post")
    ap.add_argument("--preview", metavar="CHAT_ID", help="post to this chat instead of the channel")
    ap.add_argument("--limit", type=int, default=config.MAX_STORIES)
    ap.add_argument("--lookback", type=int, default=config.LOOKBACK_HOURS)
    ap.add_argument("--plain", action="store_true", help="force the plain-text renderer")
    ap.add_argument("--no-audio", action="store_true",
                    help="skip the Persian narration (faster; text and images only)")
    ap.add_argument("--no-cover", action="store_true",
                    help="skip the drawn cover card (faster; no local image render)")
    ap.add_argument("--lookback-fixed", action="store_true",
                    help="never widen the window, even if too few stories are fresh")
    ap.add_argument("--save-payload", metavar="PATH", help="write the rich payload as JSON")
    args = ap.parse_args(argv)

    t0 = time.time()
    log = lambda m: print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

    log("fetching feeds...")
    raw = sources.collect(args.lookback)
    log(f"  {len(raw)} stories inside the {args.lookback}h window")
    if not raw:
        log("nothing to publish")
        return 0

    seen = load_seen()
    fresh = [s for s in raw if s.fingerprint not in seen]
    log(f"  {len(fresh)} unseen ({len(raw)-len(fresh)} already published)")

    # A thin bulletin is the failure the channel actually shows: post 106 shipped
    # ONE story because 8 of the 9 in the window were already published, and a
    # single-item bulletin has no gallery, no sections, nothing to read. Widen
    # the window until there is enough material rather than posting something
    # empty — older-but-unread beats fresh-but-alone.
    #
    # This must fire at zero too, not just at "thin". Measured immediately after
    # the first version shipped: the next run found 13 stories in the 8h window
    # and 0 unseen, so it went silent — while 124 unseen items sat in the 24h
    # window. Silence with a full backlog is the dead channel all over again.
    if len(fresh) < config.MIN_STORIES and not args.lookback_fixed:
        for hours in config.WIDEN_LADDER:
            if hours <= args.lookback:
                continue
            log(f"  only {len(fresh)} fresh — widening the window to {hours}h")
            raw = sources.collect(hours)
            fresh = [s for s in raw if s.fingerprint not in seen]
            log(f"  {len(fresh)} unseen in {hours}h")
            if len(fresh) >= config.MIN_STORIES:
                break

    if not fresh:
        log("no new stories — staying silent")
        return 0

    log("triaging...")
    picked = enrich.triage(fresh, args.limit)
    log(f"  kept {len(picked)}: " + ", ".join(f"{s.section}/{s.score:.0f}" for s in picked))

    log("translating (audited)...")
    ready = enrich.localise(picked)
    rejected = len(picked) - len(ready)
    log(f"  {len(ready)} passed the Persian quality gate, {rejected} rejected")
    if not ready:
        log("every translation failed the gate — refusing to post broken Persian")
        return 1

    log("writing the editor's summary...")
    summary = enrich.digest(ready)

    # Narration is optional and must never be able to block a bulletin: any
    # failure inside `audio.narrate` returns "" and the post ships text-only.
    narration_id = ""
    if not args.plain and not args.no_audio:
        log("synthesising the Persian narration...")
        narration_id = audio.narrate(
            summary, [s.title_fa for s in ready],
            args.preview or config.CHANNEL_ID,
            voice=render.current_voice())
        log("  narration attached" if narration_id else "  narration unavailable — text only")

    # The cover card: the one element of the post that carries a real colour.
    # Same optional contract as the narration — a failure returns "" and the
    # bulletin ships without it rather than not shipping.
    cover_id = ""
    if not args.plain and not args.no_cover:
        log("drawing the cover card...")
        meta = render.cover_meta(ready)
        path = card.build(**meta)
        if path:
            try:
                cover_id = telegram.upload_photo(
                    path, args.preview or config.CHANNEL_ID)
            except Exception as e:                       # noqa: BLE001
                log(f"  cover upload failed: {e}")
        log(f"  cover attached (theme {meta['theme_index']})" if cover_id
            else "  cover unavailable — bulletin ships without it")

    if args.plain:
        payload, kind = telegram.plain_fallback(ready, summary), "plain"
    else:
        payload, kind = render.build(ready, summary, narration_id, cover_id), "rich"

    if args.save_payload:
        with open(args.save_payload, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        log(f"payload written to {args.save_payload}")

    if args.dry_run:
        log(f"dry run — {kind} payload built, nothing posted")
        for s in ready:
            print(f"   [{s.section}] {s.title_fa}")
        return 0

    target = args.preview or config.CHANNEL_ID
    try:
        if kind == "rich":
            mid = telegram.send_rich(payload, target)
        else:
            mid = telegram.send_text(payload, target)
        log(f"posted message_id={mid} ({kind})")
    except telegram.TelegramError as e:
        log(f"rich send failed: {e}")
        # A photo URL that Telegram cannot fetch fails the whole call even
        # though every word of the bulletin is fine. Retry once without any
        # images before giving up on the rich format — losing the pictures is a
        # far smaller loss than dropping to plain text.
        if kind == "rich" and any(s.image for s in ready):
            log("retrying without images...")
            for s in ready:
                s.image = ""
            try:
                mid = telegram.send_rich(
                    render.build(ready, summary, narration_id, cover_id), target)
                log(f"posted message_id={mid} (rich, images dropped)")
                return _save_state(args, ready, seen)
            except telegram.TelegramError as e2:
                log(f"image-free retry also failed: {e2}")
        log("falling back to plain text...")
        mid = telegram.send_text(telegram.plain_fallback(ready, summary), target)
        log(f"posted message_id={mid} (plain fallback)")

    if not args.preview:
        now = int(datetime.now(timezone.utc).timestamp())
        for s in ready:
            seen[s.fingerprint] = now
        save_seen(seen)
        log(f"state saved ({len(seen)} fingerprints)")
    return 0


def _save_state(args, ready, seen) -> int:
    """Record the published fingerprints (skipped for previews)."""
    if args.preview:
        return 0
    now = int(datetime.now(timezone.utc).timestamp())
    for s in ready:
        seen[s.fingerprint] = now
    save_seen(seen)
    print(f"state saved ({len(seen)} fingerprints)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
