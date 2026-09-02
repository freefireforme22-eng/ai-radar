"""Offline tests — no network, no LLM. Run: python -m pytest tests/ -q"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar import config, llm, render, sources  # noqa: E402
from radar.sources import Story  # noqa: E402


def mk(title, source="Wired", tier=2, summary="", **kw):
    return Story(title_en=title, url=f"https://x/{abs(hash(title))}", source=source,
                 source_fa=source, tier=tier,
                 published=datetime.now(timezone.utc), summary_en=summary, **kw)


# ── the quality gate: the bug the user actually reported ──────────────────
def test_gate_rejects_mostly_english():
    ok, why = llm.audit("OpenAI released a new model with better reasoning هوش مصنوعی")
    assert not ok, why


def test_gate_rejects_untranslated_residue():
    text = ("شرکت OpenAI مدل جدید را منتشر کرد که در zero-shot evaluation و "
            "downstream tasks و human preference alignment و context compression "
            "و retrieval augmentation و speculative decoding بهتر است")
    ok, why = llm.audit(text)
    assert not ok and "untranslated" in why, why


def test_gate_accepts_clean_persian_with_brand_names():
    text = ("شرکت OpenAI مدل GPT-5.2 Turbo را عرضه کرد؛ مدلی که در آزمون GPQA "
            "امتیاز ۹۴.۲ درصد گرفت و هزینه پردازش را ۴۰ درصد کاهش داد.")
    ok, why = llm.audit(text)
    assert ok, why


def test_gate_rejects_empty():
    assert not llm.audit("")[0]
    assert not llm.audit("   ")[0]


def test_gate_allows_acronyms_and_versions():
    text = "این کتابخانه از طریق SDK و API در دسترس است و مدل o3 را پشتیبانی می‌کند. " * 2
    ok, why = llm.audit(text)
    assert ok, why


# ── AI relevance filter ──────────────────────────────────────────────────
def test_filter_drops_non_ai():
    assert not sources.is_ai_related(mk("Explore the Designed for Xbox Cozy Collection", "Microsoft"))
    assert not sources.is_ai_related(mk("Paint.net 5.2 alpha now runs on Linux", "Hacker News"))
    assert not sources.is_ai_related(mk("Biggest dark matter detector spots a particle", "Hacker News"))


def test_filter_keeps_ai():
    assert sources.is_ai_related(mk("NYC bans AI use for students", "The Verge"))
    assert sources.is_ai_related(mk("Introducing Gemini 3.8 Flash", "DeepMind", 1))
    assert sources.is_ai_related(mk("Anthropic ships Claude update", "Wired"))


def test_filter_bare_ai_is_token_not_substring():
    # 'said', 'maintain', 'chain' must not match the bare "ai" term
    assert not sources.is_ai_related(mk("He said the chain maintains plain rails", "Hacker News"))


def test_trusted_feeds_bypass_keywords():
    assert sources.is_ai_related(mk("Real-Time Intelligence with IBM Time Series", "Hugging Face", 1))


# ── deduplication ────────────────────────────────────────────────────────
def test_dedupe_collapses_reworded_headline():
    a = mk("Trump Administration Sides With OpenAI in New York Times Copyright Lawsuit", "Wired")
    b = mk("The Trump administration is supporting OpenAI in the NYT copyright lawsuit", "The Verge")
    out = sources.dedupe_similar([a, b])
    assert len(out) == 1
    assert out[0].also_seen_in, "the duplicate outlet should be credited"


def test_dedupe_keeps_primary_source():
    vendor = mk("Introducing Gemini 3.8 Flash and Flash Cyber", "DeepMind", tier=1)
    press = mk("Google introduces Gemini 3.8 Flash and Flash Cyber model", "The Verge", tier=2)
    out = sources.dedupe_similar([press, vendor])
    assert len(out) == 1
    assert out[0].source == "DeepMind"


def test_dedupe_keeps_distinct_stories():
    a = mk("NYC bans AI use for students until high school")
    b = mk("Amazon's AI assistant can now spot fake emails")
    assert len(sources.dedupe_similar([a, b])) == 2


# ── HTML entity decoding ─────────────────────────────────────────────────
def test_clean_text_decodes_numeric_entities():
    out = sources.clean_text("Researchers fear disaster ahead of OpenAI&#8217;s Astra release")
    assert "&#" not in out and "\u2019" in out


def test_clean_text_strips_tags():
    assert sources.clean_text("<p>Hello <b>world</b></p>") == "Hello world"


# ── rich payload structure ───────────────────────────────────────────────
def _story_ready(section="models"):
    s = mk("Introducing Gemini 3.8 Flash", "DeepMind", 1)
    s.title_fa = "معرفی مدل Gemini ۳.۸ Flash"
    s.summary_fa = "گوگل مدل جدید خود را با تمرکز بر استدلال و کاهش هزینه معرفی کرد."
    s.why_fa = "این مدل رقابت در بازار مدل‌های ارزان را شدیدتر می‌کند."
    s.facts = ["پنجره متنی بزرگ‌تر", "هزینه کمتر"]
    s.section = section
    s.score = 9
    return s


def _walk(blocks):
    for b in blocks:
        yield b
        for child in b.get("blocks", []) or []:
            yield from _walk([child])
        for item in b.get("items", []) or []:
            for child in item.get("blocks", []) or []:
                yield from _walk([child])


def test_payload_uses_only_server_accepted_types():
    # Verified live against api.telegram.org — see render.py docstring.
    accepted = {"paragraph", "heading", "pre", "footer", "divider",
                "mathematical_expression", "anchor", "list", "blockquote",
                "expandable_blockquote", "pull_quote", "collage", "slideshow",
                "table", "details", "map", "animation", "audio", "photo",
                "video", "voice_note", "thinking"}
    payload = render.build([_story_ready(), _story_ready("policy")], "جمع‌بندی آزمایشی")
    types = {b["type"] for b in _walk(payload["blocks"])}
    assert types <= accepted, types - accepted


def test_payload_is_rtl_and_headings_have_size():
    payload = render.build([_story_ready()], "")
    assert payload["is_rtl"] is True
    for b in _walk(payload["blocks"]):
        if b["type"] == "heading":
            assert isinstance(b.get("size"), int) and 1 <= b["size"] <= 6


def test_details_uses_summary_not_title():
    payload = render.build([_story_ready()], "")
    for b in _walk(payload["blocks"]):
        if b["type"] == "details":
            assert "summary" in b and "title" not in b


def test_toggles_are_nested_at_least_two_deep():
    payload = render.build([_story_ready()], "")

    def depth(blocks, d=0):
        best = d
        for b in blocks:
            if b["type"] == "details":
                best = max(best, depth(b.get("blocks", []), d + 1))
            else:
                best = max(best, depth(b.get("blocks", []), d))
        return best

    assert depth(payload["blocks"]) >= 2


def test_checklist_items_carry_checkbox():
    payload = render.build([_story_ready()], "")
    lists = [b for b in _walk(payload["blocks"]) if b["type"] == "list"]
    assert any(item.get("has_checkbox") for lst in lists for item in lst["items"])


# ── transliteration repair ────────────────────────────────────────────────
def test_translit_never_corrupts_ordinary_words():
    """The word-boundary guard: without it "پرو"→"Pro" turned "پرونده"
    (case file) into "Proنده", which reached a live preview post."""
    from radar.enrich import _repair_translit as f
    assert "پرونده" in f("دولت در پرونده نیویورک تایمز از OpenAI حمایت کرد")
    assert "Proنده" not in f("دولت در پرونده نیویورک تایمز حمایت کرد")
    assert f("پروژه جدید آغاز شد") == "پروژه جدید آغاز شد"
    assert f("سیستم پرواز خودکار") == "سیستم پرواز خودکار"
    assert f("قابلیت کلادسازی وجود ندارد") == "قابلیت کلادسازی وجود ندارد"


def test_translit_restores_brand_names():
    from radar.enrich import _repair_translit as f
    assert "DeepMind" in f("شرکت دیپ‌مایند برنامه جدیدی عرضه کرد")
    assert "Claude و OpenAI" in f("کلاد و اوپن‌ای‌آی رقیب هستند")
    assert "Gemini" in f("معرفی مدل جمینای جدید")


def test_translit_product_suffix_only_after_latin():
    from radar.enrich import _repair_translit as f
    # trailing a Latin product token -> convert
    assert f("Gemini ۳.۸ فلش سایبر منتشر شد") == "Gemini 3.8 Flash Cyber منتشر شد"
    assert "GPT-5 Turbo" in f("مدل GPT-5 توربو عرضه شد")
    # standalone Persian noun -> leave alone
    assert f("این حافظه فلش سرعت بالایی دارد") == "این حافظه فلش سرعت بالایی دارد"


def test_translit_normalises_version_digits_inside_latin_names():
    from radar.enrich import _repair_translit as f
    assert "Gemini 3.8" in f("مدل Gemini ۳.۸ عرضه شد")


# ── Jalali date ──────────────────────────────────────────────────────────
def test_jalali_known_dates():
    # Cross-checked against the jdatetime package over 1200 consecutive days:
    # zero mismatches. 2024-03-20 is Nowruz 1403, so the previous day is the
    # last day of Esfand 1402.
    assert render._jalali(datetime(2024, 3, 19, tzinfo=timezone.utc)) == "۲۹ اسفند ۱۴۰۲"
    assert render._jalali(datetime(2024, 3, 20, tzinfo=timezone.utc)) == "۱ فروردین ۱۴۰۳"
    assert render._jalali(datetime(2026, 9, 2, tzinfo=timezone.utc)) == "۱۱ شهریور ۱۴۰۵"


# ── Persian digits ───────────────────────────────────────────────────────
def test_fa_digits_preserves_model_versions():
    from radar.enrich import _fa_digits
    assert _fa_digits("مدل GPT-5.2 با ۴۰ درصد") .count("5.2") == 1
    assert "۴۰" in _fa_digits("مدل GPT-5.2 با 40 درصد")


def test_fa_digits_converts_bare_numbers():
    from radar.enrich import _fa_digits
    assert _fa_digits("94.2 درصد") == "۹۴.۲ درصد"
