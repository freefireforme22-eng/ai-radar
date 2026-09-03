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


# ── spelled-out version numbers (live defect: «نسخه صفر.سی‌وسهار» for 0.34) ──
def test_spelled_version_is_replaced_with_digits_from_source():
    from radar.enrich import _fix_spelled_version
    out, ok = _fix_spelled_version("انتشار ابزار llm-gemini نسخه صفر.سی‌وسهار", "0.34")
    assert ok
    assert "۰.۳۴" in out
    # no fragment of the spelled-out number may survive
    for frag in ("صفر", "سی", "سه", "ار"):
        assert frag not in out.split("نسخه")[1]


def test_spelled_version_without_source_number_is_rejected():
    from radar.enrich import _fix_spelled_version
    out, ok = _fix_spelled_version("ابزار نسخه صفر.سی‌وسه منتشر شد", "")
    assert ok is False, "must refuse to publish rather than invent a version"


def test_spelled_version_leaves_ordinary_prose_alone():
    from radar.enrich import _fix_spelled_version
    for text in ("این نسخه پایدار است", "نسخه خطی کتاب در موزه است", "نسخه بتا در دسترس است"):
        out, ok = _fix_spelled_version(text, "2.0")
        assert ok and out == text, text


def test_source_version_extraction():
    from radar.enrich import _source_version
    assert _source_version("llm-gemini 0.34 released", "") == "0.34"
    assert _source_version("Muse Spark v1.3 is here", "") == "1.3"
    assert _source_version("OpenAI ships a new feature", "") == ""


def test_translit_covers_every_gemini_spelling():
    from radar.enrich import _repair_translit
    for spelling in ("جمینای", "جمینی", "جیمینی", "جمنی"):
        assert "Gemini" in _repair_translit(f"مدل {spelling} معرفی شد"), spelling


# ── orthographic doubles the Persian audit cannot see ────────────────────
def test_spelling_fix_repairs_observed_typos():
    from radar.enrich import _fix_spelling
    assert _fix_spelling("دستیار آامازون") == "دستیار آمازون"
    assert _fix_spelling("گوگگل اعلام کرد") == "گوگل اعلام کرد"


def test_spelling_fix_leaves_correct_text_untouched():
    from radar.enrich import _fix_spelling
    for good in ("آمازون", "هوش مصنوعی", "گوگل", "مایکروسافت", "آینده روشن است"):
        assert _fix_spelling(good) == good, good


def test_digest_and_summary_share_the_same_repairs():
    """Every reader-visible field must go through the same cleanup chain."""
    import inspect
    from radar import enrich
    src = inspect.getsource(enrich._localise_one)
    for field in ("title_fa", "summary_fa", "why_fa"):
        line = [l for l in src.splitlines() if l.strip().startswith(f"{field} =")][0]
        assert "_fix_spelling" in line and "_repair_translit" in line, field


# ── scraped sources (Anthropic publishes no RSS at all) ──────────────────
def test_scrape_date_parses_index_format():
    from radar.sources import _scrape_date
    got = _scrape_date("Announcements Sep 1, 2026 Some headline", r"([A-Z][a-z]{2} \d{1,2}, \d{4})")
    assert got is not None and (got.year, got.month, got.day) == (2026, 9, 1)


def test_scrape_date_returns_none_when_absent():
    from radar.sources import _scrape_date
    assert _scrape_date("no date here", r"([A-Z][a-z]{2} \d{1,2}, \d{4})") is None


def test_anthropic_is_trusted_topical():
    """Anthropic ships no feed, so its stories arrive via the scraper; they must
    not then be dropped by the keyword filter for lacking an 'AI' token."""
    from radar.sources import _TRUSTED_TOPICAL
    assert "Anthropic" in _TRUSTED_TOPICAL


def test_scrape_source_config_is_well_formed():
    from radar.sources import SCRAPE_SOURCES
    for src in SCRAPE_SOURCES:
        for key in ("name", "fa", "tier", "index", "base", "link_re", "date_re"):
            assert key in src, f"{src.get('name')} missing {key}"
        assert src["index"].startswith("https://")


# ── the boundary class must exclude punctuation and digits ────────────────
def test_brand_before_persian_comma_is_repaired():
    """A live audited bulletin shipped «مدیرعامل انویدیا، و جرج کورتز».

    U+060C (Persian comma) sits inside U+0600-U+06FF, so the old boundary class
    treated it as a letter and refused to repair a name followed by a comma —
    which is where names sit constantly in Persian prose.
    """
    from radar.enrich import _repair_translit
    assert _repair_translit("مدیرعامل انویدیا، و جرج کورتز") == "مدیرعامل Nvidia، و جرج کورتز"
    assert _repair_translit("انویدیا؛ شرکت پیشرو") == "Nvidia؛ شرکت پیشرو"
    assert _repair_translit("انویدیا؟") == "Nvidia؟"
    assert _repair_translit("محصول جدید آنتروپیک، کلاد ۴") == "محصول جدید Anthropic، Claude 4"


def test_boundary_still_protects_ordinary_words():
    """The fix must not reopen the «پرونده» → «Proنده» corruption."""
    from radar.enrich import _repair_translit
    for word in ("پرونده قضایی", "پرونده\u200cهای حقوقی", "فلشبک",
                 "کلادسازی", "میسترالی", "حافظه فلش"):
        assert _repair_translit(word) == word, word


# ── Telegram's server-side linkifier ─────────────────────────────────────
def test_feed_category_is_not_turned_into_a_dead_link():
    """Live post #98 shipped a table cell reading "arXiv cs.AI"; Telegram
    rewrote it into a link whose href was the literal string "cs.AI" — a dead
    link, because .AI is a real ccTLD. Probed live: .LG is not a TLD, so only
    one of three arXiv rows broke, which is how it survived review."""
    from radar import render
    out = render.no_autolink("arXiv cs.AI")
    assert out == "arXiv cs.\u2060AI"
    assert "cs.AI" not in out


def test_no_autolink_leaves_persian_and_versions_alone():
    from radar import render
    assert render.no_autolink("\u0646\u0633\u062e\u0647 \u06f0.\u06f3\u06f4") == "\u0646\u0633\u062e\u0647 \u06f0.\u06f3\u06f4"
    assert render.no_autolink("GPT-4.5") == "GPT-4.5"          # digits after the dot
    assert render.no_autolink("Qwen3-4B") == "Qwen3-4B"        # no dot at all


def test_no_autolink_never_touches_a_deliberate_link_label():
    from radar import render
    link = render.link("cs.AI docs", "https://arxiv.org/list/cs.AI/recent")
    assert render.no_autolink(link) == link


def test_build_neutralises_host_shapes_everywhere_in_the_tree():
    """One call at the end of build() must cover table cells, checklist facts
    and the source index alike — not just the cell where it was first seen."""
    import json
    from datetime import datetime, timezone
    from radar import render
    from radar.sources import Story
    s = Story(title_en="x", url="https://arxiv.org/abs/1", source="arXiv cs.AI",
              source_fa="\u0622\u0631\u06a9\u0627\u06cc\u0648",
              published=datetime.now(timezone.utc), summary_en="y", tier=3)
    s.title_fa = "\u0639\u0646\u0648\u0627\u0646"
    s.summary_fa = "\u062e\u0644\u0627\u0635\u0647"
    s.why_fa = ""
    s.facts = ["\u0646\u06a9\u062a\u0647 cs.AI"]
    s.score = 9
    s.section = "models"
    s.also_seen_in = []
    raw = json.dumps(render.build([s], ""), ensure_ascii=False)
    assert "cs.\u2060AI" in raw
    import re
    assert not re.search(r"cs\.(?!\u2060)AI", raw)


# ── photos ────────────────────────────────────────────────────────────────
def test_photo_block_uses_the_inputmedia_shape():
    """Probed live: {"photo": "<url>"} fails with 'Field "photo" must be of type
    Object', {"photo":{"url":...}} with 'Can't find field "type"', and
    {"type":"photo","url":...} with 'media not found'. Only ``media`` works."""
    from radar import render
    b = render.photo("https://example.com/a.jpg")
    assert b == {"type": "photo",
                 "photo": {"type": "photo", "media": "https://example.com/a.jpg"}}


def test_photo_caption_must_be_an_object():
    """A bare string caption is rejected with 'RichBlockCaption must be an
    object', so the helper wraps plain text for the caller."""
    from radar import render
    assert render.photo("u", caption="\u0634\u0631\u062d")["caption"] == {"text": "\u0634\u0631\u062d"}
    rich = {"text": [{"type": "bold", "text": "x"}]}
    assert render.photo("u", caption=rich)["caption"] == rich


def test_photo_url_survives_the_autolink_guard():
    """no_autolink must not corrupt the media URL — 'example.com' inside a photo
    block is a real address, not prose."""
    import json
    from radar import render
    raw = json.dumps(render.no_autolink(render.photo("https://cdn.example.com/a.jpg")))
    assert "https://cdn.example.com/a.jpg" in raw


def test_lead_image_is_not_repeated_inside_its_own_story():
    from datetime import datetime, timezone
    from radar import render
    from radar.sources import Story
    def mk(title, img):
        s = Story(title_en="x", url=f"https://x/{title}", source="Wired",
                  source_fa="\u0648\u0627\u06cc\u0631\u062f", tier=2,
                  published=datetime.now(timezone.utc), summary_en="y")
        s.title_fa = title
        s.summary_fa = "\u062e\u0644\u0627\u0635\u0647"
        s.section = "models"
        s.score = 8
        s.image = img
        return s
    stories = [mk("\u0627\u0648\u0644", "https://img/1.jpg"), mk("\u062f\u0648\u0645", "https://img/2.jpg")]
    payload = render.build(stories, "")
    import json
    raw = json.dumps(payload)
    assert raw.count("https://img/1.jpg") == 1, "lead image duplicated inside its story"
    assert raw.count("https://img/2.jpg") == 1


# ── image validation (a dead URL kills the whole bulletin) ─────────────────
def test_verify_images_drops_unusable_urls(monkeypatch):
    """Telegram fetches photo URLs server-side and fails the ENTIRE
    sendRichMessage call on one bad link ('failed to get HTTP URL content').
    Measured live: 1 of 12 real feed images was unfetchable."""
    from datetime import datetime, timezone
    from radar import enrich
    from radar.sources import Story
    def mk(img):
        return Story(title_en="t", url="https://x", source="s", source_fa="s",
                     tier=2, published=datetime.now(timezone.utc), image=img)
    good, bad = mk("https://ok/a.jpg"), mk("https://dead/b.jpg")
    monkeypatch.setattr(enrich, "_image_is_usable", lambda u: u.startswith("https://ok"))
    enrich.verify_images([good, bad])
    assert good.image == "https://ok/a.jpg"
    assert bad.image == ""


def test_image_validator_rejects_non_image_content_type(monkeypatch):
    """The Google CMS URL that broke a real send answered 200 with
    application/octet-stream, so status alone is not enough."""
    from radar import enrich

    class FakeResp:
        status = 200
        headers = {"Content-Type": "application/octet-stream", "Content-Length": "7000"}
        def read(self, n): return b"x"
        def __enter__(self): return self
        def __exit__(self, *a): return False

    monkeypatch.setattr(enrich.urllib.request, "urlopen", lambda *a, **k: FakeResp())
    assert enrich._image_is_usable("https://storage.googleapis.com/x.webp") is False


# ── feed image extraction ─────────────────────────────────────────────────
def test_image_of_reads_all_four_carriers():
    """Probed live across 22 feeds: enclosure (VentureBeat), media:content
    (NVIDIA, Ars), media:thumbnail (Wired) and <img> in the description (Meta,
    The Verge). Checking only one carrier finds art for some and misses others."""
    import xml.etree.ElementTree as ET
    from radar import sources
    MEDIA = "http://search.yahoo.com/mrss/"

    enc = ET.fromstring('<item><enclosure url="https://a/1.jpg" type="image/jpeg"/></item>')
    assert sources._image_of(enc) == "https://a/1.jpg"

    mc = ET.fromstring(f'<item xmlns:media="{MEDIA}">'
                       f'<media:content url="https://a/2.jpg" medium="image"/></item>')
    assert sources._image_of(mc) == "https://a/2.jpg"

    th = ET.fromstring(f'<item xmlns:media="{MEDIA}">'
                       f'<media:thumbnail url="https://a/3.jpg"/></item>')
    assert sources._image_of(th) == "https://a/3.jpg"

    desc = ET.fromstring('<item><description>'
                         '&lt;img src="https://a/4.jpg"&gt;</description></item>')
    assert sources._image_of(desc) == "https://a/4.jpg"


def test_image_of_skips_tracking_pixels():
    import xml.etree.ElementTree as ET
    from radar import sources
    node = ET.fromstring('<item><description>'
                         '&lt;img src="https://feeds.feedburner.com/~ff/pixel.gif"&gt;'
                         '&lt;img src="https://cdn/real-photo.jpg"&gt;</description></item>')
    assert sources._image_of(node) == "https://cdn/real-photo.jpg"


def test_og_image_recovers_art_for_feeds_without_inline_images():
    """og:image covers 8 of the 13 feeds that ship no inline image."""
    from radar import sources
    html = '<meta property="og:image" content="https://cdn/og.jpg">'
    assert sources._og_image_from(html, "https://x") == "https://cdn/og.jpg"
    # protocol-relative
    assert sources._og_image_from('<meta property="og:image" content="//cdn/og.png">',
                                 "https://x") == "https://cdn/og.png"
    # arXiv offers only a relative path to its own logo — worse than no image
    assert sources._og_image_from(
        '<meta property="og:image" content="/static/arxiv-logo-fb.png">', "https://x") == ""


def test_fetch_article_keeps_its_string_signature():
    """enrich still calls fetch_article() in places; it must stay str-returning
    while the new tuple-returning variant does the work."""
    from radar import sources
    import inspect
    assert "tuple" in inspect.signature(sources.fetch_article_and_image).return_annotation


def test_junk_filter_drops_site_logos_from_any_path_position():
    """A real dry-run illustrated two different arXiv papers with
    static.arxiv.org/icons/twitter/arxiv-logo-twitter-square.png — the site's own
    card badge. The first version of the filter only matched a logo immediately
    before the extension, so it let this through and two distinct stories looked
    like duplicates."""
    from radar import sources
    junk = [
        "https://static.arxiv.org/icons/twitter/arxiv-logo-twitter-square.png",
        "https://site.com/assets/logo.svg",
        "https://site.com/img/favicon-256.png",
        "https://feeds.feedburner.com/~ff/pixel.gif",
    ]
    art = [
        "https://media.wired.com/photos/abc/master/pass/Security_Flock.jpg",
        "https://wp.technologyreview.com/wp-content/uploads/2026/08/MITTRI.jpg",
        "https://cdn.example.com/2026/09/deep-learning-chip.jpg",
    ]
    for u in junk:
        assert sources._JUNK_IMAGE.search(u), u
    for u in art:
        assert not sources._JUNK_IMAGE.search(u), u
