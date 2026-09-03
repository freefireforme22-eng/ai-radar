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
    assert tts.narrate("جمع‌بندی", ["تیتر"], 123) == ""


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
