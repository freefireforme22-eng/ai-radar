"""Offline tests — no network, no LLM. Run: python -m pytest tests/ -q"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radar import card, config, llm, motion, render, sources  # noqa: E402
from radar import run as run_mod  # noqa: E402
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
    s.image = "https://img.example.com/lead.jpg"
    s.impact_fa = "قیمت هر میلیون توکن برای توسعه‌دهندگان کوچک نصف می‌شود."
    s.metric_label = "دقت در MMLU"
    s.metric_value = "۹۴.۲٪"
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
    # Verified live against api.telegram.org, one block per request so a single
    # rejection cannot mask the rest. Rejected there: "map" without a nested
    # `location`, "reference" (unsupported entirely), "date_time" keyed on
    # anything but `unix_time`.
    accepted = {"paragraph", "heading", "pre", "footer", "divider",
                "mathematical_expression", "anchor", "list", "blockquote",
                "expandable_blockquote", "pullquote", "collage", "slideshow",
                "table", "details", "map", "animation", "audio", "photo",
                "video", "voice_note", "buttons", "thinking"}
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


def test_key_points_render_as_a_real_ordered_list():
    """Ordered lists key off item["type"], not "label_type".

    Round-tripped live: with `type` the stored labels are "1."/"2."/"3.", with
    `label_type` they come back as "•" — the field is accepted and silently
    dropped. The user asked for numbered lists, so this must be `type`.
    """
    payload = render.build([_story_ready()], "")
    lists = [b for b in _walk(payload["blocks"]) if b["type"] == "list"]
    numbered_items = [item for lst in lists for item in lst["items"]
                      if item.get("type") in {"1", "a", "A", "i", "I"}]
    assert numbered_items, "no ordered list in the bulletin"
    assert all("label_type" not in item for lst in lists for item in lst["items"])


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
    """Every reader-visible field must go through the same cleanup chain.

    The chain now lives in one local helper instead of being repeated per field,
    so this checks both halves: each field calls the helper, and the helper runs
    all three repairs. Previously a field could silently skip one of them —
    that is how «هاکینگ فیس» reached a live post while sibling fields were clean.
    """
    import inspect
    from radar import enrich
    src = inspect.getsource(enrich._localise_one)
    for field in ("title_fa", "summary_fa", "why_fa"):
        line = [l for l in src.splitlines() if l.strip().startswith(f"{field} =")][0]
        assert "_clean(" in line, field
    helper = src.split("def _clean(")[1].split("\n\n")[0]
    for repair in ("_fix_spelling", "_repair_translit", "_repair_brands_grounded"):
        assert repair in helper, repair
    # facts, impact and the metric label are reader-visible too.
    for other in ("facts = ", "impact = ", "story.metric_label = "):
        line = [l for l in src.splitlines() if l.strip().startswith(other)][0]
        assert "_clean(" in line, other


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
def test_bulletin_disables_server_side_entity_detection():
    """Live post #98 shipped a table cell reading "arXiv cs.AI"; Telegram
    rewrote it into a link whose href was the literal string "cs.AI" — a dead
    link, because .AI is a real ccTLD (.LG is not, which is why only one of
    three arXiv rows broke and the bug survived review).

    Probed both ways on the live API: without the flag the stored message
    carries `url` entities with url == text; with it, none. This replaced a
    U+2060 WORD JOINER hack that had to mutate every string in the tree.
    """
    payload = render.build([_story_ready()], "")
    assert payload["skip_entity_detection"] is True


def test_explicit_links_still_work_under_skip_entity_detection():
    """The flag must not cost us real links: verified live that a `url` element
    survives while bare host-shaped text stops being linkified."""
    payload = render.build([_story_ready()], "")
    urls = [t for b in _walk(payload["blocks"])
            for t in (b.get("text") if isinstance(b.get("text"), list) else [])
            if isinstance(t, dict) and t.get("type") == "url"]
    assert any(t["url"].startswith("https://") for t in urls)


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


def test_media_urls_are_never_mangled():
    """Photo/collage URLs are real addresses, not prose: nothing in the render
    path may rewrite them. (The old WORD JOINER guard had to special-case them;
    `skip_entity_detection` removes that whole class of risk.)"""
    import json
    from radar import render
    raw = json.dumps(render.build([_story_ready()], ""))
    assert "https://img.example.com/lead.jpg" in raw


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


# ── design variety (the user's complaint: "قالب پیامای الان همه‌شون شبیه همه") ──
def test_consecutive_bulletins_do_not_look_identical():
    """Four 6-hour slots must rotate through different structural themes.

    There is no colour/font field in the API (probed: `color`, `theme`, `font`
    are accepted and silently dropped), so visual identity has to come from
    rotating which quote form, gallery type and glyph set a bulletin uses.
    """
    import json
    from datetime import datetime, timezone
    from radar import render
    shapes = set()
    for hour in (1, 7, 13, 19):
        now = datetime(2026, 9, 3, hour, tzinfo=timezone.utc)
        th = render._theme(now)
        shapes.add(json.dumps(th, sort_keys=True, ensure_ascii=False))
    assert len(shapes) == 4, "two of four daily slots render the same theme"


def test_each_section_gets_its_own_list_and_quote_style():
    """Inside one bulletin, a research item must not read like a funding item."""
    from radar import render
    quotes = {v["quote"] for v in render._SECTION_STYLE.values()}
    bullets = {v["bullet"] for v in render._SECTION_STYLE.values()}
    assert len(quotes) >= 3, quotes
    assert len(bullets) >= 3, bullets


def test_photos_appear_at_top_level_not_only_inside_toggles():
    """Art buried in a collapsed toggle is invisible until tapped — which is why
    the user saw "هیچ عکسی نیست" on a bulletin that did ship two photos."""
    from radar import render
    payload = render.build([_story_ready(), _story_ready("business")], "")
    top = {b["type"] for b in payload["blocks"]}
    assert "photo" in top, top


def test_mid_message_gallery_never_repeats_a_story_photo():
    """The gallery and the per-story photos draw from the same pool, so a photo
    used in the band must be suppressed inside its own toggle."""
    import json
    from datetime import datetime, timezone
    from radar import render
    from radar.sources import Story

    def mk(i):
        s = Story(title_en=f"t{i}", url=f"https://x/{i}", source="Wired",
                  source_fa="وایرد", tier=2,
                  published=datetime.now(timezone.utc), summary_en="y")
        s.title_fa = f"تیتر {i}"
        s.summary_fa = "خلاصه"
        s.section = "models"
        s.score = 8
        s.image = f"https://img/{i}.jpg"
        return s

    payload = render.build([mk(i) for i in range(1, 5)], "")
    raw = json.dumps(payload)
    for i in range(1, 5):
        assert raw.count(f"https://img/{i}.jpg") == 1, f"image {i} duplicated"


def test_analytical_fields_are_rendered_when_present():
    """The user rejected "نکات کلیدی" that merely copied the story. The renderer
    must surface the inferred fields (impact, the key number) prominently."""
    import json
    from radar import render
    s = _story_ready()
    raw = json.dumps(render.build([s], ""), ensure_ascii=False)
    assert s.impact_fa in raw
    assert s.metric_label in raw and s.metric_value in raw
    assert "pullquote" in raw          # impact gets its own visual treatment


def test_latex_is_rendered_as_a_math_block_not_prose():
    import json
    from radar import render
    s = _story_ready()
    s.latex = r"L(N) \approx A N^{-0.34}"
    blocks = render.build([s], "")["blocks"]
    exprs = [b for b in _walk(blocks) if b["type"] == "mathematical_expression"]
    assert exprs and exprs[0]["expression"] == s.latex


def test_buttons_always_carry_a_url():
    """Probed live: a button with an empty url, an `anchor_name`, or no url at
    all is rejected with "Text buttons are not allowed in the inline keyboard",
    which fails the ENTIRE bulletin. In-post navigation must use anchor_link."""
    from radar import render
    payload = render.build([_story_ready()], "")
    for b in _walk(payload["blocks"]):
        if b["type"] == "buttons":
            for btn in b["buttons"]:
                assert btn.get("url", "").startswith("http"), btn


# ── key points must be analysis, not copied sentences ────────────────────
def test_key_points_that_merely_repeat_the_summary_are_dropped():
    """The exact complaint: "تو نکات کلیدی فقط یه چندتا جمله از تو خود خبر
    انتخاب میشه و گذاشته میشه که اصلا ارزشی نداره"."""
    from radar import enrich
    summary = "گوگل مدل جدید خود را با تمرکز بر استدلال و کاهش هزینه معرفی کرد."
    assert enrich._too_similar("گوگل مدل جدید خود را با تمرکز بر استدلال معرفی کرد", summary)
    assert not enrich._too_similar("هزینه هر میلیون توکن ۳۰ درصد کاهش یافت", summary)


def test_similarity_tolerates_shared_brand_names():
    """A key point naming the same company as the summary is not a copy."""
    from radar import enrich
    summary = "OpenAI مدل تازه‌ای برای استدلال ریاضی معرفی کرد و قیمت را کاهش داد."
    assert not enrich._too_similar("OpenAI برای نخستین بار وزن‌ها را منتشر می‌کند", summary)


def test_plain_fallback_keeps_the_analytical_fields():
    """The fallback is the emergency path; it must not silently drop the
    analysis and regress to a bare list of copied sentences."""
    from radar import telegram
    s = _story_ready()
    out = telegram.plain_fallback([s], "جمع‌بندی")
    assert s.impact_fa in out
    assert s.metric_value in out
    assert "1. " in out          # numbered, matching the rich renderer


# ── source-grounded brand repair ──────────────────────────────────────────
def test_brand_repair_is_grounded_in_the_english_source():
    """Two real live defects, both invisible to the Persian audit (100% Persian
    letters, no Latin residue): «هاکینگ فیس» for Hugging Face and «کراداستریک»
    for CrowdStrike. Neither was in TRANSLIT_FIX, which is why a fixed
    dictionary is not enough."""
    from radar.enrich import _repair_brands_grounded as fix
    out = fix("شرکت Nvidia موافقت کرده است تا پلتفرم هاکینگ فیس را خریداری کند.",
              "Nvidia agrees to buy Hugging Face in a $12.9 billion deal")
    assert "Hugging Face" in out and "هاکینگ" not in out

    out = fix("مدیرعامل کراداستریک درباره امنیت گفت.",
              "CrowdStrike CEO George Kurtz on AI security")
    # Headline case glues the job title on ("CrowdStrike CEO"); substituting the
    # glued candidate produced «مدیرعامل CrowdStrike CEO» in a real run.
    assert "CrowdStrike" in out and "CEO" not in out


def test_brand_repair_never_rewrites_ordinary_persian_prose():
    """The failure mode to avoid is the «پرونده» → «Proنده» class: a Persian word
    whose skeleton resembles a brand in the source."""
    from radar.enrich import _repair_brands_grounded as fix
    prose = "پژوهشگران این ابزار را بررسی کردند و پرونده جدیدی گشودند."
    assert fix(prose, "Perplexity launches a research agent") == prose


def test_brand_repair_leaves_persian_household_names_alone():
    """گوگل/آمازون are standard in Persian journalism; forcing them to Latin
    makes the text read worse, and the user asked for Persian."""
    from radar.enrich import _repair_brands_grounded as fix
    prose = "گوگل اعلام کرد که مدل تازه را منتشر می‌کند."
    assert fix(prose, "Google DeepMind ships Gemini upgrades") == prose


def test_brand_repair_cannot_invent_a_name_absent_from_the_source():
    """Only names present in this story's own English text are candidates."""
    from radar.enrich import _repair_brands_grounded as fix
    txt = "شرکت کراداستریک محصول جدیدی معرفی کرد."
    assert fix(txt, "Microsoft ships a new Windows build") == txt


def test_velar_and_soft_c_collapse_in_the_skeleton():
    """Persian transliteration swaps ک/گ freely and "Face" ends in /s/, not /k/;
    without both rules «هاکینگ فیس» scored 0.67 against "Hugging Face" and the
    live defect went unrepaired."""
    from difflib import SequenceMatcher
    from radar.enrich import _skeleton_fa, _skeleton_latin
    ratio = SequenceMatcher(None, _skeleton_latin("Hugging Face"),
                            _skeleton_fa("هاکینگفیس")).ratio()
    assert ratio >= 0.82


def test_headline_case_words_are_not_treated_as_brands():
    """Measured on live message 5058: headline case capitalises every word, so
    "Safety Awareness Benchmark" made the repair fire on ordinary Persian —
    «این بنچمارک» → «این Benchmark», «۱۸۰۰ پیکسل» → «۱۸۰۰ PCs», «سال جاری» →
    «سال Garrett» — dropping the bulletin from 98.7% to 93.8% Persian."""
    from radar.enrich import _latin_brands, _repair_brands_grounded as fix
    assert _latin_brands("A new Safety Awareness Benchmark for deployment") == []

    unchanged = "این بنچمارک با هر ارزیابی سازگار کار می‌کند و واریانس را می‌سنجد."
    assert fix(unchanged, "A new Safety Awareness Benchmark for deployment") == unchanged

    px = "نمایشگر اولد با رزولوشن ۲۸۸۰ در ۱۸۰۰ پیکسل است."
    assert fix(px, "Lenovo Yoga 9n has a 16-inch OLED at 2880 by 1800 pixels") == px

    yr = "شرکت Flock اعلام کرده تا پایان سال جاری قوانین را اعمال می‌کند."
    assert fix(yr, "Flock CEO Garrett Langley said rules arrive by year end") == yr


def test_two_word_brands_still_repair_after_the_vocabulary_filter():
    """"Safety" alone is vocabulary, but "Flock Safety" is a company: the phrase
    survives because one of its parts qualifies."""
    from radar.enrich import _repair_brands_grounded as fix
    out = fix("شرکت فلاک سیفتی شبکه دوربین را گسترش داد.",
              "Flock Safety expands camera network across Texas")
    assert "Flock Safety" in out


def test_short_skeletons_are_rejected_as_too_ambiguous():
    """Measured on live message 5061: "Garrett" reduces to the 3-consonant
    skeleton KRT, which scores 1.00 against «کارت» (card) and 0.86 against
    «کارت‌های», and the bulletin shipped «تا پایان سال جاری میلادی Garrett
    اجباری». Skeletons shorter than 5 consonants are not discriminating."""
    from radar.enrich import _repair_brands_grounded as fix
    src = ("Flock Safety says CEO Garrett Langley will require case numbers "
           "and automatic audits by year end")
    for txt in ("تا پایان سال جاری میلادی کارت اجباری اعمال می‌شود.",
                "شرکت کارت‌های پرونده را اجباری می‌کند."):
        assert fix(txt, src) == txt


def test_long_skeleton_brands_still_repair():
    """The 5-consonant floor must not cost any real repair."""
    from radar.enrich import _repair_brands_grounded as fix
    wins = [
        ("Nvidia agrees to buy Hugging Face", "پلتفرم هاکینگ فیس", "Hugging Face"),
        ("CrowdStrike CEO on security", "شرکت کراداستریک", "CrowdStrike"),
        ("Flock Safety expands cameras", "شرکت فلاک سیفتی", "Flock Safety"),
        ("Mistral raises funding", "شرکت میسترال", "Mistral"),
    ]
    for src, txt, want in wins:
        assert want in fix(txt, src), want


def test_curated_brands_match_spelling_variants():
    """The live defect «هاکینگ فیس» was one letter away from the dictionary entry
    «هاگینگ فیس» (گ vs ک), which is why an exact-string table missed it. Curated
    brands are matched by skeleton so any variant of a KNOWN brand is caught."""
    from radar.enrich import _repair_brands_grounded as fix
    for variant in ("هاکینگ فیس", "هاگینگ فیس", "هاگینگفیس"):
        assert "Hugging Face" in fix(f"پلتفرم {variant} فروخته شد", "")


def test_source_derived_names_use_a_stricter_floor():
    """Live 5063: «مقیاس بزرگ‌تر» (BSRKTR) matched "Abstract" (BSTRKT) at 0.83,
    so open-ended source names need a higher floor than curated ones."""
    from radar.enrich import _repair_brands_grounded as fix, _CURATED_MIN_RATIO, _GROUNDED_MIN_RATIO
    assert _GROUNDED_MIN_RATIO > _CURATED_MIN_RATIO
    txt = "فریب مدل‌های با مقیاس بزرگ‌تر می‌شود."
    assert fix(txt, "Abstract We study Qwen3 models. Larger models deceived.") == txt


def test_common_english_stragglers_are_translated():
    """Live 5065 shipped «ماه September» and «فناوری Rendering عصبی». The audit
    counts Latin *words*, so one or two per sentence always pass — exactly the
    "mostly Persian with English sprinkled in" outcome the user rejected."""
    from radar.enrich import _translate_stragglers as tr
    assert "سپتامبر" in tr("سرویس در ماه September میزبان بازی است.")
    assert "رندرینگ" in tr("بازی از فناوری Rendering عصبی بهره می‌برد.")


def test_straggler_translation_never_breaks_a_proper_name():
    """"Visual Concepts" and "GeForce NOW" are names: a capitalised Latin
    neighbour means the word belongs to the name and must stay."""
    from radar.enrich import _translate_stragglers as tr
    for keep in ("این فناوری نتیجه همکاری Nvidia با استودیوهای Visual Concepts است.",
                 "سرویس GeForce NOW فعال است.",
                 "شرکت Flock Safety دوربین‌ها را گسترش داد."):
        assert tr(keep) == keep


def test_clean_helper_runs_the_straggler_pass_too():
    import inspect
    from radar import enrich
    src = inspect.getsource(enrich._localise_one)
    helper = src.split("def _clean(")[1].split("\n\n")[0]
    assert "_translate_stragglers" in helper


# ── narrated edition (answers "خیلی خشک و خالیه فقط متنه") ────────────────
def test_narration_block_uses_the_probed_audio_shape():
    """`audio` + file_id in an InputMedia object, with a caption that survives.

    Probed live: `voice_note` is also accepted but SILENTLY DROPS the caption,
    so the narration would arrive unlabelled. `audio` keeps it."""
    payload = render.build([_story_ready()], "جمع‌بندی", "FILEID123")
    blocks = [b for b in _walk(payload["blocks"]) if b["type"] == "audio"]
    assert len(blocks) == 1
    a = blocks[0]
    assert a["audio"] == {"type": "audio", "media": "FILEID123"}
    assert a["caption"]["text"], "narration must be labelled"


def test_bulletin_without_narration_is_unchanged():
    """TTS must never be able to break a bulletin: no file_id, no audio block,
    everything else identical."""
    quiet = render.build([_story_ready()], "جمع‌بندی")
    loud = render.build([_story_ready()], "جمع‌بندی", "FILEID123")
    assert not [b for b in _walk(quiet["blocks"]) if b["type"] == "audio"]
    assert len(loud["blocks"]) == len(quiet["blocks"]) + 1


def test_narration_script_is_persian_and_bounded():
    from radar import audio as tts
    script = tts.narration_text("جمع‌بندی سردبیر امروز.",
                                ["تیتر اول", "تیتر دوم", "تیتر سوم",
                                 "تیتر چهارم", "تیتر پنجم", "تیتر ششم"],
                                clock="۱۲:۳۰")
    assert script.startswith("رادار هوش مصنوعی.")
    assert "تیتر پنجم" in script and "تیتر ششم" not in script, "at most 5 headlines"
    assert len(script) <= 1800
    import re as _re
    latin = _re.findall(r"[A-Za-z]{2,}", script)
    assert not latin, f"narration script must be Persian, found {latin}"


def test_narrator_voice_rotates_with_the_theme():
    from radar import audio as tts
    voices = {render._theme(datetime(2026, 9, 4, h, 0)) ["voice"] for h in (1, 7, 13, 19)}
    assert len(voices) >= 2, "consecutive bulletins must not all use one voice"
    assert voices <= set(tts.VOICES)


def test_narration_failure_returns_empty_string(monkeypatch):
    """Every failure path must degrade to "" rather than raising."""
    from radar import audio as tts
    monkeypatch.setattr(tts, "synthesise", lambda *a, **k: "")
    assert tts.narrate("جمع‌بندی", ["تیتر"], 123) == ("", "")


# ── per-story block variety (map / citation) ──────────────────────────────
def test_arxiv_story_gets_a_monospace_citation_card():
    """`pre` finally earns its place: a BibTeX card for a paper, which no other
    story in the bulletin carries."""
    from radar import geo
    s = _story_ready("models")
    s.url = "https://arxiv.org/abs/2509.04321"
    s.title_en = "Sparse Kernels for Long Context"
    s.citation = geo.citation(s.url, s.title_en, "arXiv cs.LG")
    payload = render.build([s], "")
    pres = [b for b in _walk(payload["blocks"]) if b["type"] == "pre"]
    assert pres, "arXiv story must carry a citation card"
    assert "2509.04321" in pres[0]["text"]
    assert pres[0]["language"] == "bibtex"


def test_non_arxiv_story_has_no_citation_card():
    payload = render.build([_story_ready("business")], "")
    assert not [b for b in _walk(payload["blocks"]) if b["type"] == "pre"]


def test_map_block_matches_the_probed_shape():
    """Probed live: `location` object is required; a flat lat/lon is rejected."""
    s = _story_ready("policy")
    s.map_lat, s.map_lon, s.map_label = 50.8467, 4.3525, "بروکسل"
    payload = render.build([s], "")
    maps = [b for b in _walk(payload["blocks"]) if b["type"] == "map"]
    assert len(maps) == 1
    assert maps[0]["location"] == {"latitude": 50.8467, "longitude": 4.3525}
    assert maps[0]["caption"]["text"][1] == "بروکسل"


def test_only_one_story_per_bulletin_gets_a_map():
    """Three maps in a row would be the same monotony, in map form. `decorate`
    stops after the first hit."""
    from radar import enrich, geo
    calls = []

    def fake_locate(text):
        calls.append(text)
        return (1.0, 2.0, "جایی")

    original = geo.locate
    geo.locate = fake_locate
    try:
        stories = [_story_ready("policy") for _ in range(3)]
        for st in stories:
            st.summary_en = "The European Union said..."
        enrich.decorate(stories)
    finally:
        geo.locate = original
    assert len(calls) == 1, "must stop geocoding after the first match"
    assert sum(1 for st in stories if st.map_lat) == 1


def test_geocoder_budget_is_capped():
    """A runaway bulletin must not hammer Nominatim: the module enforces its own
    lookup ceiling."""
    from radar import geo
    geo.reset()
    assert geo._MAX_LOOKUPS <= 3
    geo._lookups = geo._MAX_LOOKUPS
    assert geo._geocode("Brussels, Belgium") is None, "budget must block lookups"
    geo.reset()


def test_citation_is_skipped_for_a_bare_arxiv_listing_url():
    from radar import geo
    assert geo.citation("https://arxiv.org/list/cs.AI/recent", "X", "arXiv") == ""


def test_hybrid_persian_latin_words_are_repaired():
    """Live 5119 shipped «خودregressive» (from "autoregressive"): the model
    translated the prefix and kept the English stem. The audit is blind to it —
    the token is mostly Persian, so the Latin *word* count barely moves."""
    from radar.enrich import _translate_stragglers as tr
    assert tr("تولید خودregressive توکن‌ها") == "تولید خودبازگشتی توکن‌ها"
    assert tr("یادگیری خودsupervised") == "یادگیری خودنظارت‌شده"
    assert tr("مدل چندmodal") == "مدل چندوجهی"


def test_unknown_hybrid_tail_is_dropped_not_shipped():
    """An unrecognised Latin tail glued to Persian is residue either way; keeping
    the Persian head is the lesser evil."""
    from radar.enrich import _translate_stragglers as tr
    assert "fineturning" not in tr("روش خودfineturning دارد")


def test_hybrid_repair_leaves_real_names_and_persian_alone():
    from radar.enrich import _translate_stragglers as tr
    for safe in ("Hugging Face و OpenAI مدل دادند", "دقت مدل GPT-5 بالاست",
                 "ترنسفورمر و رمزگذار سالم بمانند"):
        assert tr(safe) == safe


def test_theme_rotation_reaches_every_theme_at_a_fixed_hour():
    """The real bug this guards: with four themes and four 6-hour slots,
    `yday * 4 + slot` is congruent to `slot` mod 4, so the 06:00 bulletin was 🛰
    every single day forever. The multiplier must stay coprime with the theme
    count."""
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 1, 1, 6, 30, tzinfo=timezone.utc)
    marks = {render._theme(base + timedelta(days=d))["mark"] for d in range(30)}
    assert len(marks) == len(render._THEMES), (
        f"a reader checking at 06:00 only ever sees {marks}")


def test_consecutive_bulletins_never_share_a_theme():
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)
    seq = [render._theme(base + timedelta(hours=6 * i))["mark"] for i in range(60)]
    assert all(a != b for a, b in zip(seq, seq[1:]))


def test_sections_have_genuinely_different_heading_sizes():
    """They were all size 3, which made "each section reads differently" false."""
    sizes = {k: v["size"] for k, v in render._SECTION_STYLE.items()}
    assert len(set(sizes.values())) == len(sizes), sizes


def test_every_section_bullet_kind_is_a_documented_label_type():
    documented = {"1", "a", "A", "i", "I"}
    for key, style in render._SECTION_STYLE.items():
        assert style["bullet"] in documented, key


# ── thin-bulletin guard (channel post 106 shipped ONE story) ──────────────
def _fake_story(fp, hours_old=1):
    """`fingerprint` is a read-only property derived from the url, so vary the
    url and read the fingerprint back instead of assigning it."""
    from datetime import timedelta
    return Story(
        title_en=f"Story {fp}", url=f"https://example.com/{fp}", source="Test",
        source_fa="تست", tier=1, published=datetime.now(timezone.utc) - timedelta(hours=hours_old),
    )


def test_window_widens_when_almost_everything_was_already_published(monkeypatch, tmp_path):
    """Channel post 106: 9 stories in the window, 8 already seen, so the
    bulletin shipped a single item -- no sections, no gallery, nothing to read.
    The pipeline must widen the window instead of publishing something thin."""
    from radar import run as run_mod

    calls = []

    def fake_collect(hours):
        calls.append(hours)
        # The 8h window holds 3 stories, 2 of them already published; the wider
        # windows hold progressively more unseen material.
        if hours <= 8:
            return [_fake_story("old1"), _fake_story("old2"), _fake_story("new1")]
        return [_fake_story("old1"), _fake_story("old2"),
                _fake_story("new1"), _fake_story("new2"),
                _fake_story("new3"), _fake_story("new4")]

    monkeypatch.setattr(run_mod.sources, "collect", fake_collect)
    monkeypatch.setattr(run_mod, "load_seen",
                        lambda: {_fake_story("old1").fingerprint: 1,
                                 _fake_story("old2").fingerprint: 1})
    monkeypatch.setattr(run_mod.enrich, "triage", lambda st, keep=None: st[:keep or 9])
    monkeypatch.setattr(run_mod.enrich, "localise", lambda st: st)
    monkeypatch.setattr(run_mod.enrich, "digest", lambda st: "جمع‌بندی")

    rc = run_mod.main(["--dry-run", "--no-audio", "--lookback", "8"])
    assert rc == 0
    assert calls[0] == 8, "the configured window must be tried first"
    assert len(calls) > 1, "a thin window must trigger a widen"
    assert calls[1] in config.WIDEN_LADDER


def test_a_healthy_window_is_never_widened(monkeypatch):
    """The widen must be a fallback, not a habit: enough fresh material means
    exactly one fetch, so the bulletin stays as fresh as the cadence allows."""
    from radar import run as run_mod

    calls = []

    def fake_collect(hours):
        calls.append(hours)
        return [_fake_story(f"s{i}") for i in range(7)]

    monkeypatch.setattr(run_mod.sources, "collect", fake_collect)
    monkeypatch.setattr(run_mod, "load_seen", lambda: {})
    monkeypatch.setattr(run_mod.enrich, "triage", lambda st, keep=None: st[:keep or 9])
    monkeypatch.setattr(run_mod.enrich, "localise", lambda st: st)
    monkeypatch.setattr(run_mod.enrich, "digest", lambda st: "جمع‌بندی")

    run_mod.main(["--dry-run", "--no-audio", "--lookback", "8"])
    assert calls == [8], f"expected a single fetch, got {calls}"


def test_lookback_fixed_opts_out_of_widening(monkeypatch):
    """A preview asking for an exact window must get that window, so measuring
    the real cadence stays possible."""
    from radar import run as run_mod

    calls = []
    monkeypatch.setattr(run_mod.sources, "collect",
                        lambda h: (calls.append(h), [_fake_story("only")])[1])
    monkeypatch.setattr(run_mod, "load_seen", lambda: {})
    monkeypatch.setattr(run_mod.enrich, "triage", lambda st, keep=None: st)
    monkeypatch.setattr(run_mod.enrich, "localise", lambda st: st)
    monkeypatch.setattr(run_mod.enrich, "digest", lambda st: "جمع‌بندی")

    run_mod.main(["--dry-run", "--no-audio", "--lookback", "8", "--lookback-fixed"])
    assert calls == [8]


def test_every_theme_layout_places_every_segment_exactly_once():
    """A typo in a layout tuple would silently DROP the narration or the hero
    image from that slot's bulletin — one post in six missing its audio, which
    is exactly the kind of silent loss that shipped photos into a collapsed
    toggle for weeks."""
    expected = {"hero", "digest", "audio", "motion", "nav", "board", "gallery"}
    for th in render._THEMES:
        assert set(th["layout"]) == expected, f"{th['mark']} layout {th['layout']}"
        assert len(th["layout"]) == len(expected), f"{th['mark']} repeats a segment"


def test_themes_differ_in_the_opening_sequence_not_just_decoration():
    """Rotating quote shapes and dividers left the block ORDER identical across
    themes, so posts still looked alike at a glance (the user's «مو نمیزنه»).
    The first blocks a reader sees must differ per theme."""
    openings = {tuple(th["layout"][:3]) for th in render._THEMES}
    assert len(openings) == len(render._THEMES), (
        f"only {len(openings)} distinct openings for {len(render._THEMES)} themes")


def test_headline_board_heading_size_varies_by_theme():
    """The board was hardcoded at size 5 for every slot."""
    assert len({th["board_size"] for th in render._THEMES}) >= 3
    for th in render._THEMES:
        assert 1 <= th["board_size"] <= 6, "documented heading sizes are 1..6"


def test_bulletin_renders_with_every_theme_layout():
    """Guards the segment-assembly loop: a layout naming a segment that build()
    never creates must not crash or produce an empty bulletin."""
    stories = [_story_ready("models"), _story_ready("business"), _story_ready("policy")]
    for s in stories:
        s.image = "https://example.com/a.jpg"
    for th in render._THEMES:
        blocks = []
        for name in th["layout"]:
            blocks.append(name)
        assert blocks, th["mark"]
    payload = render.build(stories, "جمع‌بندی", "FAKE_FILE_ID")
    kinds = {b.get("type") for b in payload["blocks"]}
    assert "audio" in kinds and "photo" in kinds


def test_zero_fresh_stories_still_widens_before_going_silent():
    """Measured right after the widen shipped: 13 stories in the 8h window, 0
    unseen -> the run went silent, while 124 unseen items sat in the 24h window.
    Guarding zero separately because `if fresh and ...` reads as correct."""
    import inspect
    from radar import run as run_mod
    src = inspect.getsource(run_mod.main)
    guard = [l for l in src.splitlines() if "config.MIN_STORIES" in l][0]
    assert "fresh and" not in guard, (
        "the widen must also run when the fresh count is zero: " + guard.strip())


def test_backfill_respects_a_section_ceiling():
    """Post 110 shipped 6 of 9 stories from `models` (eight arXiv cards): the
    backfill loop ignored MAX_PER_SECTION outright, so any widened window --
    which is mostly arXiv -- became a research digest."""
    from radar import enrich
    from datetime import timedelta
    pool = []
    for i in range(12):
        s = Story(title_en=f"Paper {i}", url=f"https://arxiv.org/abs/26{i:02d}.1",
                  source="arXiv cs.AI", source_fa="آرکایو", tier=3,
                  published=datetime.now(timezone.utc) - timedelta(hours=1))
        s.section = "models"
        s.score = 9 - i * 0.1
        pool.append(s)
    for i in range(2):
        s = Story(title_en=f"Deal {i}", url=f"https://example.com/biz{i}",
                  source="Test", source_fa="تست", tier=1,
                  published=datetime.now(timezone.utc) - timedelta(hours=1))
        s.section = "business"
        s.score = 5
        pool.append(s)

    picked = enrich._spread(pool, keep=9)
    models = sum(1 for s in picked if s.section == "models")
    ceiling = config.MAX_PER_SECTION + config.BACKFILL_SLACK
    assert models <= ceiling, f"{models} models items exceed the ceiling {ceiling}"


def test_citation_cards_are_capped_per_bulletin():
    """Eight BibTeX cards in post 110: a block on every story is wallpaper, and
    'every story looks the same' is the complaint this feature was meant to fix."""
    stories = []
    for i in range(6):
        s = _story_ready("models")
        s.citation = f"@misc{{x{i}}}"
        stories.append(s)
    payload = render.build(stories, "جمع‌بندی")
    cards = sum(1 for b in _walk(payload["blocks"]) if b.get("type") == "pre")
    assert cards <= render._MAX_CITATIONS, f"{cards} citation cards rendered"


def test_one_source_family_cannot_fill_the_bulletin():
    """Post 112: balanced across all four sections, nine arXiv abstracts, zero
    photos. arXiv papers are classified into every section, so a section cap
    alone does not stop a research-only bulletin -- and arXiv carries no art."""
    from radar import enrich
    from datetime import timedelta
    ranked = []
    for i in range(20):
        s = Story(title_en=f"Paper {i}", url=f"https://arxiv.org/abs/26{i:02d}.9",
                  source=f"arXiv cs.{'AI' if i % 3 == 0 else 'LG'}",
                  source_fa="آرکایو", tier=3,
                  published=datetime.now(timezone.utc) - timedelta(hours=1))
        s.section = ["models", "business", "policy", "tools"][i % 4]
        s.score = 8
        ranked.append(s)
    for i in range(6):
        s = Story(title_en=f"News {i}", url=f"https://example.com/n{i}",
                  source="The Verge", source_fa="ورج", tier=1,
                  published=datetime.now(timezone.utc) - timedelta(hours=1))
        s.section = ["models", "business", "policy", "tools"][i % 4]
        s.score = 7
        ranked.append(s)
    for i in range(6):
        s = Story(title_en=f"Wired {i}", url=f"https://example.com/w{i}",
                  source="Wired", source_fa="وایرد", tier=1,
                  published=datetime.now(timezone.utc) - timedelta(hours=1))
        s.section = ["models", "business", "policy", "tools"][i % 4]
        s.score = 7
        ranked.append(s)

    picked = enrich._spread(ranked, keep=9)
    arxiv = sum(1 for s in picked if s.source_fa == "آرکایو")
    ceiling = config.MAX_PER_FAMILY + config.BACKFILL_SLACK
    assert arxiv <= ceiling, f"{arxiv} arXiv items of {len(picked)} (ceiling {ceiling})"
    assert len(picked) == 9, f"the bulletin must still be filled, got {len(picked)}"


def test_a_full_bulletin_beats_balance_when_the_pool_is_one_sided():
    """The caps must not zero out the bulletin: on a night when only arXiv
    published, ship MIN_STORIES research items rather than three slots and a
    lot of white space — but NOT nine, which is what posts 110/112 did."""
    from radar import enrich
    from datetime import timedelta
    ranked = []
    for i in range(20):
        s = Story(title_en=f"Paper {i}", url=f"https://arxiv.org/abs/27{i:02d}.9",
                  source="arXiv cs.AI", source_fa="آرکایو", tier=3,
                  published=datetime.now(timezone.utc) - timedelta(hours=1))
        s.section = ["models", "business", "policy", "tools"][i % 4]
        s.score = 8
        ranked.append(s)
    picked = enrich._spread(ranked, keep=9)
    assert len(picked) == config.MIN_STORIES, (
        f"a one-sided pool should stop at MIN_STORIES, got {len(picked)}")


def test_non_persian_arabic_script_letters_are_folded():
    """The live dry run produced «بڈراک» for Bedrock -- U+0688 is Urdu, not
    Persian, and the Persian-RATIO audit is blind to it because the character
    still counts as Arabic script. Folding keeps the word readable."""
    from radar.enrich import _fix_spelling
    assert "ڈ" not in _fix_spelling("آمازون بڈراک")
    assert _fix_spelling("كتاب مصنوعي") == "کتاب مصنوعی"   # Arabic kaf + yeh
    assert _fix_spelling("مدل هوش مصنوعی") == "مدل هوش مصنوعی"  # untouched


def test_the_digest_runs_through_the_spelling_pipeline():
    """The editor's summary sits at the top of the bulletin and used to bypass
    `_fix_spelling` entirely, so a stray foreign letter shipped unrepaired."""
    import inspect
    from radar import enrich
    src = inspect.getsource(enrich.digest)
    assert "_fix_spelling" in src, "digest() must normalise the script too"


# ── «نکات کلیدی» substance filter ─────────────────────────────────────────
def test_vacuous_key_points_are_rejected():
    """The exact filler that shipped on channel post 114: novel wording, zero
    summary overlap, and nothing a reader could use."""
    from radar import facts as facts_mod
    filler = [
        "این نسخه عملکرد سایبری گوگل را تقویت می‌کند.",
        "تمرکز اصلی مدل جدید بر انجام وظایف استدلالی بلندمدت است.",
        "گروه‌های تروریستی در حال حاضر از هوش مصنوعی استفاده می‌کنند.",
        "این آموزش شامل مراحل گام‌به‌گام راه‌اندازی سراسری سیستم است.",
        "یک برنامه سایبری مشابه سایر رقبای هوش مصنوعی معرفی شده است.",
    ]
    for f in filler:
        assert not facts_mod.has_substance(f), f"filler passed the substance test: {f}"


def test_substantive_key_points_survive():
    """Points that actually shipped and were worth reading must not be lost."""
    from radar import facts as facts_mod
    real = [
        "قیمت توکن‌های ورودی و خروجی به ترتیب ۱.۲۵ و ۴.۲۵ دلار به ازای هر میلیون توکن است.",
        "نسخه max در بنچمارک τ³-Banking با کسب ۵۲ درصد رتبه نخست را به دست آورد.",
        "بیش از ۱۸ میلیون توسعه‌دهنده از هاب Hugging Face استفاده می‌کنند.",
        "عربستان قصد دارد تا سال ۲۰۳۴ میلادی شش گیگابایت دیتاسنتر مستقر کند.",
    ]
    for f in real:
        assert facts_mod.has_substance(f), f"real point was dropped: {f}"


def test_every_kept_point_gets_a_kind_label():
    """The label drives the per-point glyph, so a kept point with no kind would
    render as an unmarked line while its siblings are marked."""
    from radar import facts as facts_mod
    for f in ["مدل جدید ۷۰ میلیارد پارامتر دارد.",
              "این نسخه نسبت به قبلی ۲ برابر سریع‌تر است.",
              "محدودیت اصلی مصرف حافظه است.",
              "تا سال ۲۰۲۷ عرضه می‌شود."]:
        assert facts_mod.primary_kind(f), f"kept point has no kind: {f}"


def test_a_bare_comparative_without_a_number_is_filler():
    """«نسبت به قبلی سریع‌تر است» is not a comparison a reader can check, and it
    passed the first version of the filter on the keyword alone."""
    from radar import facts as facts_mod
    assert not facts_mod.has_substance("این نسخه نسبت به قبلی سریع‌تر است.")
    assert facts_mod.has_substance("این نسخه نسبت به قبلی ۴۰ درصد سریع‌تر است.")


def test_kind_marks_cover_every_kind_the_filter_can_return():
    """A kind with no glyph silently falls back to a generic mark, which is the
    'every point looks the same' problem in miniature."""
    from radar import facts as facts_mod
    kinds = {name for name, _ in facts_mod._KINDS}
    assert kinds <= set(render._KIND_MARK), (
        f"kinds without a glyph: {kinds - set(render._KIND_MARK)}")


def test_key_points_render_with_distinct_marks():
    import json as _json
    s = _story_ready("models")
    s.facts = ["مدل ۷۰ میلیارد پارامتر دارد.",
               "محدودیت اصلی مصرف حافظه است.",
               "تا سال ۲۰۲۷ عرضه می‌شود."]
    payload = render.build([s], "جمع‌بندی")
    dumped = _json.dumps(payload, ensure_ascii=False)
    marks = [m for m in render._KIND_MARK.values() if m in dumped]
    assert len(marks) >= 2, f"only {len(marks)} distinct kind marks rendered"


def test_leaked_category_prefix_is_stripped():
    """The salvage prompt names the five categories, and the model echoed one:
    «محدودیت یا ریسک: بسیاری از ارائه‌دهندگان…» shipped in a live dry run."""
    from radar import facts as facts_mod
    cases = [
        ("محدودیت یا ریسک: کاهش هزینه‌ها پروژه‌های میلیارد دلاری را تهدید می‌کند.",
         "کاهش هزینه‌ها پروژه‌های میلیارد دلاری را تهدید می‌کند."),
        ("عدد مشخص: قیمت هر میلیون توکن ۴.۲۵ دلار است.",
         "قیمت هر میلیون توکن ۴.۲۵ دلار است."),
        ("مقایسه — این مدل ۲ برابر سریع‌تر است.", "این مدل ۲ برابر سریع‌تر است."),
    ]
    for raw, want in cases:
        assert facts_mod.strip_label(raw) == want, facts_mod.strip_label(raw)


def test_a_normal_point_is_not_mangled_by_prefix_stripping():
    from radar import facts as facts_mod
    f = "بیش از ۱۸ میلیون توسعه‌دهنده از این پلتفرم استفاده می‌کنند."
    assert facts_mod.strip_label(f) == f


def test_vague_growth_verbs_need_a_number():
    """«قابلیت‌های عملکردی نسبت به قبل تقویت شده است» passed the first version of
    the filter on the keyword «نسبت به» while saying nothing measurable."""
    from radar import facts as facts_mod
    assert not facts_mod.has_substance(
        "قابلیت‌های عملکردی در حوزه سایبری نسبت به قبل تقویت شده است.")
    assert facts_mod.has_substance(
        "عملکرد نسبت به نسخه قبل ۳۰ درصد تقویت شده است.")


def test_keywords_buried_inside_longer_words_do_not_count():
    """Persian has no case and few delimiters, so `in` matching produces false
    positives that cannot be spotted by eye. Live post 118 labelled
    «گزارش‌ها حاکی از دریافت ایمیل‌های متعدد…» as ⚠️ risk because «دریافت»
    contains «افت» (loss)."""
    from radar import facts as facts_mod
    assert not facts_mod.has_substance(
        "گزارش‌ها حاکی از دریافت ایمیل‌های متعدد توسط محققان است.")
    # ... while the real word still matches, including with a suffix.
    assert facts_mod.has_substance("افت شدید فروش تراشه در این فصل ثبت شد.")


def test_a_magnitude_word_without_a_number_is_not_a_magnitude():
    """«کاربران هوش مصنوعی» hit the scale list while quantifying nothing —
    shipped on live post 118 as 📈."""
    from radar import facts as facts_mod
    assert not facts_mod.has_substance(
        "این کمپین علیه کاربران هوش مصنوعی اجرا شده بود.")
    assert facts_mod.has_substance("۱۸ میلیون کاربر از این سرویس استفاده می‌کنند.")


def test_version_digits_alone_are_not_a_fact():
    """«تحت لایسنس Apache 2.0 منتشر شده‌اند» is a licence name, not a number the
    reader learns anything from. It shipped on post 118 marked 📊."""
    from radar import facts as facts_mod
    assert not facts_mod.has_substance(
        "تمام مدل‌ها و کدهای این مجموعه تحت لایسنس Apache 2.0 منتشر شده‌اند.")


def test_rate_limiting_is_a_feature_not_a_risk():
    """«محدودیت نرخ» is rate limiting. Post 118 rendered it with the risk glyph."""
    from radar import facts as facts_mod
    assert not facts_mod.has_substance(
        "پشتیبانی از بودجه‌بندی و محدودیت نرخ برای درخواست‌ها")
    assert facts_mod.has_substance("محدودیت اصلی این روش مصرف حافظه است.")


# ── the cover card (the only colour in the post) ───────────────────────────
def test_cover_palette_per_theme_is_distinct():
    """«اگر هر پیام رنگ و ویژگی خاص خودشو داشته باشه بهتر میشه» — no rich block
    type carries a colour, so colour arrives as pixels. Two themes sharing a
    palette would put the channel back to «همه شبیه هم» on the axis the user
    actually notices first."""
    from radar import card, render
    assert len(card.PALETTES) == len(render._THEMES)
    tops = [p[0] for p in card.PALETTES]
    assert len(set(tops)) == len(tops), "two themes share a background colour"
    accents = [p[2] for p in card.PALETTES]
    assert len(set(accents)) == len(accents), "two themes share an accent"


def test_cover_theme_index_matches_the_text_theme():
    """The card and the bulletin must never disagree about which theme it is —
    two independent rotations would drift and the post would look assembled from
    two different designs."""
    from datetime import datetime, timedelta, timezone
    from radar import render
    start = datetime(2026, 1, 1, 0, 5, tzinfo=timezone.utc)
    for day in range(40):
        for hour in (0, 6, 12, 18):
            now = start + timedelta(days=day, hours=hour)
            assert render._THEMES[render.theme_index(now)] is render._theme(now)


def test_cover_sits_after_the_heading_not_before():
    """The channel post title comes from the FIRST heading block. A cover photo
    placed above it costs the post its title."""
    from radar import render
    s = _story_ready("models")
    payload = render.build([s], "خلاصه", "", "FAKE_FILE_ID")
    types = [b["type"] for b in payload["blocks"]]
    assert types.index("heading") < types.index("photo")
    assert payload["blocks"][types.index("photo")]["photo"]["media"] == "FAKE_FILE_ID"


def test_bulletin_ships_without_a_cover():
    """Every enrichment is optional by contract: a failed render must not cost
    the channel its bulletin."""
    from radar import render
    s = _story_ready("models")
    payload = render.build([s], "خلاصه", "", "")
    media = [b for b in payload["blocks"] if b["type"] == "photo"]
    assert all(b["photo"]["media"] != "" for b in media)


def test_cover_meta_carries_three_headlines_and_persian_digits():
    from radar import render
    stories = [_story_ready("models"), _story_ready("business"),
               _story_ready("policy"), _story_ready("tools")]
    meta = render.cover_meta(stories)
    assert set(meta) == {"theme_index", "date_fa", "clock", "count_fa", "headlines"}
    assert len(meta["headlines"]) == 3, "the card has room for exactly three"
    assert meta["count_fa"] == "۴", "Latin digits on a Persian card"
    assert 0 <= meta["theme_index"] < len(render._THEMES)


def test_card_build_is_a_noop_when_dependencies_are_missing(monkeypatch):
    """CI installs Pillow and the shaper; a future runner without them must
    degrade rather than crash the publish step."""
    from radar import card
    monkeypatch.setattr(card, "available", lambda: False)
    assert card.build(theme_index=0, date_fa="۱", clock="۱۲:۰۰",
                      count_fa="۳", headlines=["الف"]) == ""


def test_persian_on_the_card_is_reshaped_and_reordered():
    """PIL on the runner has raqm=False/harfbuzz=False: no complex-script
    shaping. Without pre-shaping, every card would carry disconnected,
    left-to-right letters that still LOOK like text at thumbnail size."""
    import pytest
    pytest.importorskip("arabic_reshaper")
    pytest.importorskip("bidi")
    from radar import card
    raw = "میلیارد"
    shaped = card._shape(raw)
    assert shaped != raw, "text went to the renderer unshaped"
    # presentation forms live in the U+FB50..U+FEFF Arabic Presentation Forms blocks
    assert any(0xFB50 <= ord(ch) <= 0xFEFF for ch in shaped)
    assert shaped[0] != raw[0], "visual order was not reversed for RTL"


def test_thin_triage_result_widens_the_window_too():
    """Post 125: 6 unseen stories, three scored 0 as promo round-ups, bulletin
    shipped THREE items — 13 block types and no gallery band. The fresh-count
    guard passed (6 >= MIN_STORIES) and the thinning happened downstream, so the
    ladder must also re-check what survived triage."""
    import radar.run as run_mod
    from radar import config

    pool_8h = [_fake_story(f"a{i}", hours_old=1) for i in range(6)]
    pool_24h = pool_8h + [_fake_story(f"b{i}", hours_old=20) for i in range(14)]

    calls = []

    def fake_collect(hours):
        calls.append(hours)
        return list(pool_24h if hours > config.LOOKBACK_HOURS else pool_8h)

    def fake_triage(stories, keep):
        # the scorer rejects everything from the narrow pool but likes the wider one
        good = [s for s in stories if s.url.startswith("https://x/b")]
        return good[:keep] if good else stories[:3]

    monkey = _Monkey()
    monkey.set(run_mod.sources, "collect", fake_collect)
    monkey.set(run_mod.enrich, "triage", fake_triage)
    monkey.set(run_mod, "load_seen", lambda: {})
    monkey.set(run_mod.enrich, "localise", lambda picked: picked)
    monkey.set(run_mod.enrich, "digest", lambda ready: "خلاصه")
    try:
        rc = run_mod.main(["--dry-run", "--no-audio", "--no-cover"])
    finally:
        monkey.undo()
    assert rc == 0
    assert any(h > config.LOOKBACK_HOURS for h in calls), \
        f"never widened after a thin triage; collect() called with {calls}"


class _Monkey:
    """Tiny setattr/undo helper: these tests patch module attributes and must
    restore them even on failure, since the whole suite shares one import."""

    def __init__(self):
        self._saved = []

    def set(self, obj, name, value):
        self._saved.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._saved):
            setattr(obj, name, old)


def test_story_shapes_differ_inside_one_bulletin():
    """Post 128 measured: seven story cards, all assembled in the same order —
    scrolling one bulletin looked like the same card seven times. Neighbouring
    cards must not share an arrangement."""
    from datetime import datetime, timezone
    from radar import render

    stories = []
    for n in range(6):
        s = _story_ready("models")
        s.title_fa = f"خبر {n}"
        s.image = f"https://example.com/{n}.jpg"
        s.metric_label, s.metric_value = "ارزش", "۱۰ میلیارد دلار"
        s.why_fa = "چون بازار را عوض می‌کند."
        s.facts = ["مدل ۷۰ میلیارد پارامتر دارد."]
        stories.append(s)

    payload = render.build(stories, "جمع‌بندی")

    def order(blocks):
        return [b["type"] for b in blocks]

    cards = []
    for sec in [b for b in payload["blocks"] if b["type"] == "details"]:
        for st in [b for b in (sec.get("blocks") or []) if b["type"] == "details"]:
            cards.append(order(st["blocks"]))

    assert len(cards) >= 4, cards
    for a, b in zip(cards, cards[1:]):
        assert a != b, f"two adjacent cards share an arrangement: {a}"
    assert len({tuple(c) for c in cards}) >= 3, cards


def test_every_story_shape_keeps_every_part_exactly_once():
    """A typo in one shape's order tuple would silently DROP a story's picture or
    its key points — the same class of bug as the layout tuples."""
    from radar import render
    s = _story_ready("models")
    s.image = "https://example.com/a.jpg"
    s.metric_label, s.metric_value = "دقت", "۹۴٪"
    s.why_fa, s.impact_fa = "مهم است.", "عوض می‌شود."
    s.facts = ["مدل ۷۰ میلیارد پارامتر دارد."]
    s.citation = "@misc{x}"
    s.latex = "E=mc^2"
    s.map_lat, s.map_lon, s.map_label = 1.0, 2.0, "جایی"

    style = render._SECTION_STYLE["models"]
    baseline = None
    for shape in render._STORY_SHAPES:
        got = render._story_blocks(s, 1, style, shape=shape)
        counts = {}
        for b in got:
            counts[b["type"]] = counts.get(b["type"], 0) + 1
        if baseline is None:
            baseline = counts
        assert counts == baseline, f"{shape} changed the parts, not just the order"


def test_every_live_feed_url_is_not_in_dead_feeds():
    """Three sweeps in, a rejected URL could easily get re-added by copy-paste.
    DEAD_FEEDS exists to record the observed reason; it is worthless if a URL can
    sit in both lists."""
    from radar import config
    dead = set(config.DEAD_FEEDS)
    for feed in config.FEEDS:
        assert feed["url"] not in dead, f"{feed['name']} is in DEAD_FEEDS: {config.DEAD_FEEDS.get(feed['url'])}"


def test_feed_list_has_enough_non_arxiv_families():
    """Post 131: 5 stories, ONE photo, from a 93-item unseen pool that was 92
    arXiv. The caps cannot invent variety that the source list does not have, so
    the picture-carrying families are the real dependency."""
    from radar import config
    fams = {f["fa"] for f in config.FEEDS if f["tier"] < 3}
    assert len(fams) >= 20, sorted(fams)


def test_digit_prefixed_units_stay_ascii():
    """Live post 134 shipped «۲K و ۴K»: _VERSION_RUN starts at a LETTER, so the
    leading digit of a resolution was converted on its own and left a
    mixed-script token inside one word — «خودregressive» in the other order.
    Numbers that are plain quantities must still convert."""
    from radar.enrich import _fa_digits
    assert "2K" in _fa_digits("وضوح تصویر: 2K و 4K")
    assert "4K" in _fa_digits("وضوح تصویر: 2K و 4K")
    assert "5G" in _fa_digits("شبکه 5G")
    assert "3D" in _fa_digits("نمایش 3D")
    assert "4x" in _fa_digits("پردازنده 4x سریع‌تر")
    # ... while real quantities are still localised
    assert "۱۲" in _fa_digits("قیمت 12 میلیارد دلار")
    assert "۱۵۰۰" in _fa_digits("سرعت 1500 توکن بر ثانیه")
    assert "۹۴.۲" in _fa_digits("دقت 94.2 درصد")
    # and version suffixes keep working
    assert "GPT-5.2" in _fa_digits("مدل GPT-5.2 منتشر شد")
    # A DECIMAL prefix, from live post 141 («رزولوشن ۲.8K»): reaching back over
    # one digit run left the "2" outside the protected span and split the token
    # one character further left.
    assert "2.8K" in _fa_digits("نمایشگر ۱۴ اینچی با رزولوشن 2.8K")
    assert "۶.۷" in _fa_digits("نمایشگر 6.7 اینچی")  # not a unit: still localised


def test_animated_chart_counts_match_the_bulletin():
    """The moving chart must count exactly what the text sections contain — a
    chart that disagrees with the post below it is worse than no chart."""
    stories = [
        _story_ready("models"), _story_ready("models"),
        _story_ready("business"), _story_ready("tools"),
    ]
    rows = render.section_counts(stories)
    assert sum(n for _, n in rows) == len(stories)
    assert [n for _, n in rows] == [2, 1, 1], rows
    # empty sections must not appear as zero-length bars
    assert all(n > 0 for _, n in rows)
    labels = dict(config.SECTIONS)
    assert [lbl for lbl, _ in rows] == [labels["models"], labels["business"],
                                        labels["tools"]]


def test_animation_block_rides_a_file_id_and_every_theme_places_it():
    """`animation` is the only block that moves; it takes a file_id like photo
    and audio, so no public hosting is needed. Every theme must give it a slot,
    otherwise a theme would silently drop the chart it just rendered."""
    block = render.animation("FILEID123", caption="نمودار")
    assert block["type"] == "animation"
    assert block["animation"] == {"type": "animation", "media": "FILEID123"}
    assert block["caption"] == {"text": "نمودار"}
    for th in render._THEMES:
        assert "motion" in th["layout"], th["mark"]

    payload = render.build([_story_ready()], "خلاصه", "AUDIOID", "COVERID", "MOTIONID")
    kinds = [b["type"] for b in payload["blocks"]]
    assert "animation" in kinds, kinds
    # and it must be absent when no chart was produced
    plain = render.build([_story_ready()], "خلاصه", "AUDIOID", "COVERID", "")
    assert "animation" not in [b["type"] for b in plain["blocks"]]


def test_motion_loop_is_a_full_size_mp4_with_moving_frames():
    """Three things this must never regress:
    1. it MOVES (frames differ — otherwise it is a still image with extra bytes),
    2. it is an MP4, because Telegram downscales uploaded GIFs to 320px wide and
       Persian labels at that size are unreadable,
    3. it keeps the authored 720x420, verified by decoding the file itself."""
    if not card.available():
        import pytest
        pytest.skip("Pillow/fonts unavailable")
    path = motion.build(theme_index=0, rows=[("الف", 3), ("ب", 1)],
                        out_path="/tmp/test_motion.mp4")
    assert path, "no loop produced"
    assert path.endswith(".mp4"), f"fell back to {path} — GIF gets downscaled"

    import subprocess as sp
    import imageio_ffmpeg
    exe = imageio_ffmpeg.get_ffmpeg_exe()
    probe = sp.run([exe, "-hide_banner", "-i", path], capture_output=True, text=True)
    assert "720x420" in probe.stderr, probe.stderr[-400:]

    # decode two distant frames as raw grayscale and compare
    def frame(n):
        out = sp.run([exe, "-loglevel", "error", "-i", path, "-vf",
                      f"select=eq(n\\,{n})", "-vframes", "1", "-f", "rawvideo",
                      "-pix_fmt", "gray", "-"], capture_output=True)
        return sum(out.stdout)
    assert frame(0) != frame(12), "frames identical — not an animation"


def test_coverage_checklist_ticks_only_sections_with_stories():
    """The checklist carries real information: an EMPTY section is invisible in
    the headline board, so this is the only place a reader learns the slot had no
    policy news. Ticked must mean "covered" — never decoration. It must also be
    ONE checklist near the headlines, not a second copy buried in the closed
    sources toggle at the bottom (where the original was effectively unseen)."""
    stories = [_story_ready("models"), _story_ready("tools")]
    payload = render.build(stories, "خلاصه")
    lists = [b for b in payload["blocks"] if b["type"] == "list"
             and any(i.get("has_checkbox") for i in b["items"])]
    assert len(lists) == 1, "expected exactly one checklist"
    items = lists[0]["items"]
    assert len(items) == len(config.SECTIONS)
    checked = {i["blocks"][0]["text"] for i in items if i.get("is_checked")}
    unchecked = {i["blocks"][0]["text"] for i in items if not i.get("is_checked")}
    labels = dict(config.SECTIONS)
    assert labels["models"] in checked and labels["tools"] in checked
    assert labels["policy"] in unchecked, unchecked
    assert all(i.get("has_checkbox") for i in items)


def test_entity_dedupe_collapses_a_wire_story_retold_by_four_outlets():
    """Live 8h pool, verbatim headlines: the same acquisition arrived four times
    with a top pairwise Jaccard of 0.33 — under the 0.42 keyword threshold — and
    three copies shipped in one bulletin. Shared entity names are what identify
    the event."""
    titles = [
        "Nvidia confirms $12.9B acquisition of AI hosting platform Hugging Face",
        "Nvidia acquires Hugging Face for $12.93 billion — company gains control",
        "With Hugging Face Acquisition, Nvidia Scores Big Win in AI Race",
        "Nvidia buys Hugging Face, the GitHub of AI, for $13 billion",
    ]
    pool = [mk(t, f"Outlet{i}", tier=i + 1) for i, t in enumerate(titles)]
    out = sources.dedupe_similar(pool)
    assert len(out) == 1, [s.title_en for s in out]
    assert len(out[0].also_seen_in) == 3, out[0].also_seen_in

    # and it must NOT collapse two unrelated stories that merely share one name
    a = mk("Nvidia launches free GPU clustering utility", "A")
    b = mk("Nvidia RTX Spark N1X launches in October", "B")
    assert len(sources.dedupe_similar([a, b])) == 2


def test_garbled_persian_is_rejected_even_though_it_is_all_persian():
    """A mangled word is invisible to both existing gates: it is Persian script,
    so the ratio reads 1.00, and it carries no Latin residue. A dry run shipped
    «ططمیع یک میلیارد دلاری…» as a headline. Persian never doubles ط at the start
    of a word."""
    ok, why = llm.audit("ططمیع یک میلیارد دلاری OpenAI برای امنیت خدمات حیاتی")
    assert not ok and "garbled" in why, why
    ok, _ = llm.audit("تطمیع یک میلیارد دلاری OpenAI برای امنیت خدمات حیاتی")
    assert ok

    # Real Persian words DO start with doubled م / ب — those must stay legal, or
    # the gate would reject correct translations.
    for good in ("ممکن است این مدل ارزان‌تر باشد و هزینه را کاهش دهد",
                 "ببرند این فناوری را به بازارهای تازه و رقابت را بیشتر کنند",
                 "ممیزی امنیتی مدل تازه منتشر شد و نتایج آن عمومی است"):
        ok, why = llm.audit(good)
        assert ok, f"{good} -> {why}"


def test_paraphrased_key_points_are_dropped_as_echo():
    """The complaint «نکات کلیدی ... چند جمله از تو خود خبر» survived the trigram
    gate because a paraphrase shares almost no trigrams with its source. Measured
    over 222 facts from live posts 118-141: 11 repeated 80%+ of their own story's
    content words while `_too_similar` flagged ZERO of them.

    Real pair from live post 128 (trigram overlap 22%): the summary already says
    the state firm admitted being three years behind Neuralink."""
    from radar.enrich import _echoes, _too_similar
    summary = ("چین امسال مجوز تجهیزات مختلف رابط مغز و رایانه را صادر کرده و با "
               "یارانه‌های دولتی از استارتاپ‌های داخلی حمایت می‌کند. با این حال، یکی از "
               "شرکت‌های پیشروی دولتی اذعان کرده که فناوری این کشور هنوز حدود سه سال "
               "از Neuralink عقب‌تر است.")
    echo = "یک شرکت پیشروی دولتی در چین اعتراف کرد فناوری‌اش سه سال از Neuralink عقب‌تر است."
    assert not _too_similar(echo, summary), "trigram gate must be the one that MISSES this"
    assert _echoes(echo, summary), "containment gate must catch it"


def test_a_fact_with_a_new_number_survives_the_echo_gate():
    """Containment alone would throw away additive points. Live post 128 shipped
    «۵ مدل پشتیبان» at 78% word overlap with a figure absent from the summary —
    trading a repetition defect for a data-loss defect is not a fix."""
    from radar.enrich import _echoes
    summary = "این پژوهش عاملی برای ارزیابی مدل‌های زبانی معرفی می‌کند و روی چند مدل آزمایش شده است."
    additive = "در این پژوهش از ۵ مدل پشتیبان مختلف شامل مدل‌های متن‌باز و بسته استفاده شده است."
    assert not _echoes(additive, summary)
    # ... but repeating a number the summary already gave is still an echo
    summary_with_num = "سرعت این مدل به ۱۵۰۰ توکن بر ثانیه می‌رسد و برای بارهای سنگین بهینه شده است."
    assert _echoes("سرعت پردازش این مدل به ۱۵۰۰ توکن بر ثانیه می‌رسد.", summary_with_num)


def test_short_points_are_left_to_the_other_gates():
    """A 3-word point cannot be judged by word overlap without false positives;
    `has_substance` and the length check own that case."""
    from radar.enrich import _echoes
    assert not _echoes("قیمت ۶۹۹ دلار", "قیمت پایه این لپ‌تاپ ۶۹۹ دلار است و در نوامبر عرضه می‌شود.")


def test_narration_block_kind_rotates_with_the_theme():
    """A voice bubble and an audio attachment are two different reading
    experiences; using only one forever is the monotony the rotation exists to
    break. Both kinds must be reachable, and the block type must follow the
    theme's declared kind — not the other way round."""
    kinds = {t.get("narration", "audio") for t in render._THEMES}
    assert kinds == {"audio", "voice_note"}, kinds

    stories = [_story_ready()]
    for kind, expected in (("voice_note", "voice_note"), ("audio", "audio")):
        payload = render.build(stories, "جمع‌بندی", narration_id="FILEID",
                               narration_kind=kind)
        types = [b["type"] for b in payload["blocks"]]
        assert expected in types, (kind, types)
        assert ("audio" if expected == "voice_note" else "voice_note") not in types


def test_voice_note_keeps_its_caption():
    """Re-probed live (messages 5269/5270): an older comment in this repo claimed
    `caption` is dropped on `voice_note`. Telegram stored it. The builder must
    therefore emit one, or the narration ships unlabelled."""
    block = render.voice_note("FILEID", caption="روایت صوتی")
    assert block["type"] == "voice_note"
    assert block["voice_note"]["media"] == "FILEID"
    assert block["caption"]["text"] == "روایت صوتی"


def test_half_transliterated_names_are_rejected():
    """Live post 148 shipped «Clement-جونز» for Clement-Jones — a proper name
    translated halfway. It defeats the ratio gate (one short fragment) and the
    Latin-residue gate (the Latin half is a legal proper noun), the same way
    «خودregressive» did in the other direction.

    The two false-positive classes measured across 1,346 live strings must keep
    passing: a Latin word followed by a Persian comma («OpenAI،»), which appears
    in 11 of 15 live posts, and Persian plural/possessive suffixes attached to
    Latin acronyms («APIها», «LLMهای»)."""
    from radar.llm import audit, _half_transliterated

    ok, reason = audit("Lord Tim Clement-جونز اصلاحیه‌ای بر لایحه امنیت سایبری پیشنهاد کرد.")
    assert not ok and "half-transliterated" in reason, reason
    assert audit("Lord Tim Clement-Jones اصلاحیه‌ای بر لایحه امنیت سایبری پیشنهاد کرد.")[0]

    # legitimate: Persian punctuation after a Latin name
    assert audit("شرکت OpenAI، گوگل و Anthropic هر سه سرمایه‌گذاری کرده‌اند و رقابت شدید است.")[0]
    # legitimate: Persian suffix on a Latin acronym
    assert audit("APIهای جدید و LLMهای بازمتن در این نسخه پشتیبانی می‌شوند و کارایی دارند.")[0]
    assert _half_transliterated("APIها و LLMهای بازمتن") == []


def test_drawn_cards_only_replace_missing_photography():
    """Real feed art must always win; the drawn card is a fallback, never a
    substitute. Measured need: live posts 141/148/152 had 0/4, 1/9 and 1/6 story
    cards carrying any picture."""
    from datetime import datetime, timezone
    from radar import render
    from radar.sources import Story

    def mk(title, img):
        s = Story(title_en="x", url=f"https://x/{title}", source="Wired",
                  source_fa="\u0648\u0627\u06cc\u0631\u062f", tier=2,
                  published=datetime.now(timezone.utc), summary_en="y")
        s.title_fa, s.summary_fa, s.image = title, "\u062e\u0644\u0627\u0635\u0647", img
        return s

    stories = [mk("\u0627\u0644\u0641", "https://img/a.jpg"), mk("\u0628", ""), mk("\u067e", "")]
    specs = render.story_card_specs(stories)
    assert [i for i, _ in specs] == [1, 2], "only the art-less stories get a card"
    # Palettes are keyed off rank so neighbouring cards differ inside one post.
    assert specs[0][1]["palette"] != specs[1][1]["palette"]
    assert specs[0][1]["rank_fa"] == "\u06f2"


def test_a_drawn_card_reaches_the_story_toggle():
    """The card has to end up in the SAME place a photo would, or the reader
    still sees a wall of text."""
    import json
    from datetime import datetime, timezone
    from radar import render
    from radar.sources import Story
    s = Story(title_en="x", url="https://x/1", source="arXiv", source_fa="\u0622\u0631\u06a9\u0627\u06cc\u0648",
              tier=1, published=datetime.now(timezone.utc), summary_en="y")
    s.title_fa, s.summary_fa = "\u062a\u06cc\u062a\u0631", "\u062e\u0644\u0627\u0635\u0647"
    s.card = "DRAWN_FILE_ID"
    raw = json.dumps(render.build([s], "\u062c\u0645\u0639"))
    assert "DRAWN_FILE_ID" in raw


def test_story_card_text_stays_inside_the_card():
    """A title long enough to wrap must not run past the margins or over the
    footer. The vision service is unavailable, so this is measured in pixels."""
    import os
    import pytest
    from radar import card
    if not card.available():
        pytest.skip("Pillow or the Persian shaping stack is unavailable")
    from PIL import Image
    long_fa = ("\u0637\u0631\u062d \u0627\u06cc\u062c\u0627\u062f \u06a9\u0644\u06cc\u062f \u062a\u0648\u0642\u0641 "
               "\u0627\u0636\u0637\u0631\u0627\u0631\u06cc \u0647\u0648\u0634 \u0645\u0635\u0646\u0648\u0639\u06cc " * 6)
    path = card.build_story(rank=3, rank_fa="\u06f3", section_fa="\u0633\u06cc\u0627\u0633\u062a",
                            title_fa=long_fa, source_fa="\u06af\u0627\u0631\u062f\u06cc\u0646",
                            metric="\u06f3 \u0628\u0631\u0627\u0628\u0631", palette=2)
    assert path
    try:
        img = Image.open(path).convert("RGB")
        assert img.size == (card.SW, card.SH)
        ink = card.PALETTES[2][3]
        px = img.load()
        xs = [x for y in range(0, card.SH, 3) for x in range(card.SW)
              if all(abs(px[x, y][i] - ink[i]) < 40 for i in range(3))]
        assert xs, "the title drew no ink at all"
        assert min(xs) >= 40 and max(xs) <= card.SW - 50
    finally:
        os.unlink(path)


def test_rotation_pin_overrides_the_clock():
    """The clock formula collapsed under real publish times.

    Measured on the 22 real publishes in the git history of data/seen.json:
    17 of 21 consecutive pairs shipped the SAME theme because several bulletins
    land inside one six-hour slot. The pin must win over the clock so the step
    happens once per published bulletin instead.
    """
    from datetime import datetime
    render.set_rotation(None)
    try:
        clock = render.theme_index(datetime(2026, 9, 3, 20, 10))
        for i in range(render.theme_count()):
            render.set_rotation(i)
            assert render.theme_index(datetime(2026, 9, 3, 20, 10)) == i
        render.set_rotation(None)
        assert render.theme_index(datetime(2026, 9, 3, 20, 10)) == clock
    finally:
        render.set_rotation(None)


def test_rotation_pin_reaches_every_theme_and_never_repeats_consecutively():
    """A counter, unlike the clock, cannot give two posts in a row one theme."""
    n = render.theme_count()
    seq = [(i + 1) % n for i in range(-1, 3 * n)]
    assert set(seq) == set(range(n)), "counter must reach every theme"
    assert all(a != b for a, b in zip(seq, seq[1:])), "no consecutive repeats"


def test_rotation_pin_drives_cover_and_motion_together(tmp_path, monkeypatch):
    """Cover, motion chart and text must agree, or the post looks assembled
    from two different designs."""
    from datetime import datetime
    render.set_rotation(2)
    try:
        assert render.theme_index(datetime(2026, 1, 1, 0, 0)) == 2
        meta = render.cover_meta([_story_ready()])
        assert meta["theme_index"] == 2
    finally:
        render.set_rotation(None)


def test_rotation_survives_a_round_trip_through_disk(tmp_path, monkeypatch):
    """A counter that does not persist is the clock bug with extra steps."""
    path = tmp_path / "rotation.json"
    monkeypatch.setattr(config, "ROTATION_PATH", str(path))
    assert run_mod.load_rotation() == -1        # missing file
    run_mod.save_rotation(4)
    assert run_mod.load_rotation() == 4
    path.write_text("{ not json", encoding="utf-8")
    assert run_mod.load_rotation() == -1        # corrupt file must not crash
    run_mod.save_rotation(render.theme_count() + 1)
    assert run_mod.load_rotation() == 1         # wraps modulo the theme count
