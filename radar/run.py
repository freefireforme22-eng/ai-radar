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

from . import config, enrich, render, sources, telegram


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

    if args.plain:
        payload, kind = telegram.plain_fallback(ready, summary), "plain"
    else:
        payload, kind = render.build(ready, summary), "rich"

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


if __name__ == "__main__":
    sys.exit(main())
