"""Turn raw English stories into audited Persian editorial content."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from . import config, llm
from .sources import Story, fetch_article_and_image

SYSTEM = (
    "تو سردبیر یک کانال خبری فارسی‌زبان در حوزه هوش مصنوعی هستی. "
    "فارسی روان، دقیق و حرفه‌ای می‌نویسی؛ نه ترجمه تحت‌اللفظی، نه لحن تبلیغاتی. "
    "هرگز چیزی از خودت به خبر اضافه نمی‌کنی."
)

_SECTION_KEYS = {name for name, _ in config.SECTIONS}

TRIAGE_PROMPT = """این فهرست عنوان‌های خبری حوزه هوش مصنوعی است. مهم‌ترین‌ها را برای یک بولتن فارسی انتخاب کن.

معیار انتخاب:
- خبر واقعی و تازه (انتشار مدل، سرمایه‌گذاری، قانون، ابزار مهم) امتیاز بالا
- تحلیل عمیق و پژوهش شاخص امتیاز متوسط
- محتوای تبلیغاتی، لیست‌های تکراری، «۱۰ ابزاری که...» امتیاز صفر

برای هر خبر یک شیء JSON بده با این کلیدها:
  i: شماره خبر
  score: عدد ۰ تا ۱۰
  section: یکی از models / business / policy / tools

فقط یک آرایه JSON برگردان، بدون توضیح.

خبرها:
{items}"""

STORY_PROMPT = """این خبر را برای یک کانال فارسی‌زبان هوش مصنوعی بازنویسی کن.

عنوان اصلی: {title}
منبع: {source}
متن: {body}

یک شیء JSON با این کلیدها برگردان:
  "title_fa":   عنوان فارسی، حداکثر ۹ کلمه، بدون علامت تعجب
  "summary_fa": خلاصه فارسی در ۲ تا ۳ جمله کامل
  "why_fa":     یک جمله: چرا این خبر مهم است
  "facts":      آرایه‌ای از ۲ تا ۴ نکته کوتاه فارسی (هر کدام حداکثر ۱۲ کلمه)

قواعد قطعی:
- تمام متن باید فارسی باشد. هیچ کلمه انگلیسی ترجمه‌نشده باقی نماند.
- تنها استثنا: نام شرکت‌ها، مدل‌ها و معیارها به لاتین بماند (OpenAI, GPT-5, Gemini, MMLU).
- نام‌های خاص را با حروف فارسی نویسه‌گردانی نکن. «OpenAI» بنویس، نه «اوپن‌ای‌آی».
- اعداد را با ارقام فارسی بنویس: ۹۴.۲ درصد
- هیچ ادعایی که در متن اصلی نیست اضافه نکن. اگر متن ناقص است، فقط از عنوان استفاده کن.
- فقط JSON برگردان."""


def triage(stories: list[Story], keep: int | None = None) -> list[Story]:
    """Score and classify with one LLM call, then keep the best."""
    keep = keep or config.MAX_STORIES
    if not stories:
        return []
    pool = stories[:60]
    listing = "\n".join(
        f"{i}. [{s.source}] {s.title_en}" for i, s in enumerate(pool)
    )
    try:
        verdicts = llm.json_call(TRIAGE_PROMPT.format(items=listing),
                                 config.MODEL_FAST, system=SYSTEM, max_tokens=2000)
    except llm.LLMError:
        # Deterministic fallback: trust tier order, guess the section by keyword.
        for s in pool:
            s.score = 10 - s.tier
            s.section = _guess_section(s.title_en)
        return pool[:keep]

    if isinstance(verdicts, dict):
        verdicts = verdicts.get("items") or verdicts.get("results") or []

    for v in verdicts:
        if not isinstance(v, dict):
            continue
        try:
            idx = int(v.get("i", -1))
        except (TypeError, ValueError):
            continue
        if not 0 <= idx < len(pool):
            continue
        story = pool[idx]
        try:
            story.score = float(v.get("score", 0))
        except (TypeError, ValueError):
            story.score = 0.0
        section = str(v.get("section", "")).strip().lower()
        story.section = section if section in _SECTION_KEYS else _guess_section(story.title_en)

    ranked = [s for s in pool if s.score > 0]
    if not ranked:                        # model returned nothing usable
        for s in pool:
            s.score = 10 - s.tier
            s.section = _guess_section(s.title_en)
        ranked = pool
    ranked.sort(key=lambda s: (-s.score, s.tier))

    # Spread across sections so one topic can't fill the whole bulletin.
    picked: list[Story] = []
    per: dict[str, int] = {}
    for s in ranked:
        if per.get(s.section, 0) >= config.MAX_PER_SECTION:
            continue
        picked.append(s)
        per[s.section] = per.get(s.section, 0) + 1
        if len(picked) >= keep:
            break
    for s in ranked:                      # backfill if sections were sparse
        if len(picked) >= keep:
            break
        if s not in picked:
            picked.append(s)
    return picked


_SECTION_HINTS = {
    "business": ("funding", "raise", "raises", "valuation", "ipo", "acquire", "acquisition",
                 "revenue", "investment", "billion", "million", "startup", "deal"),
    "policy":   ("regulation", "law", "act", "ban", "court", "lawsuit", "policy", "senate",
                 "eu ", "compliance", "safety institute", "executive order", "copyright"),
    "tools":    ("release", "open source", "open-source", "sdk", "api", "library",
                 "framework", "cli", "plugin", "available now", "launch"),
}


def _guess_section(title: str) -> str:
    low = title.lower()
    for section, hints in _SECTION_HINTS.items():
        if any(h in low for h in hints):
            return section
    return "models"


# Brand/model names the models transliterate into Persian letters despite the
# instruction. Observed live: a title reading "جمینای ۳.۸ فلش" whose own body
# correctly said "Gemini 3.8 Flash" — one headline rendered two ways. Repaired
# deterministically because the model is inconsistent *within a single response*.
#
# Only names where transliteration is actually wrong or confusing are listed.
# Company names that are standard in Persian journalism (گوگل، مایکروسافت،
# آمازون) are deliberately left alone — forcing those to Latin makes the text
# read worse, and the user asked for Persian, not for Latin sprinkled about.
TRANSLIT_FIX = {
    # Gemini has many plausible Persian spellings and the model picks a
    # different one per call; a live bulletin shipped "جمینی" because only the
    # "-ای" spellings were listed. Enumerate every variant seen in the wild.
    "جمینای": "Gemini", "جمنای": "Gemini", "جیمینای": "Gemini",
    "جمینی": "Gemini", "جمنی": "Gemini", "جیمینی": "Gemini",
    "جمیني": "Gemini", "جِمینای": "Gemini", "جِمینی": "Gemini",
    "اوپن‌ای‌آی": "OpenAI", "اوپن ای آی": "OpenAI", "اپن‌ای‌آی": "OpenAI",
    "اوپن‌ای": "OpenAI", "اپن ای آی": "OpenAI",
    "چت‌جی‌پی‌تی": "ChatGPT", "چت جی پی تی": "ChatGPT",
    "جی‌پی‌تی": "GPT", "جی پی تی": "GPT",
    "آنتروپیک": "Anthropic", "انتروپیک": "Anthropic",
    "کلاد": "Claude", "کلود": "Claude",
    "دیپ‌مایند": "DeepMind", "دیپ مایند": "DeepMind",
    "دیپ‌سیک": "DeepSeek", "دیپ سیک": "DeepSeek",
    "لاما": "Llama", "میسترال": "Mistral", "کوئن": "Qwen", "گراک": "Grok",
    "کوپایلوت": "Copilot", "کو‌پایلوت": "Copilot",
    "هاگینگ‌فیس": "Hugging Face", "هاگینگ فیس": "Hugging Face",
    "میدجرنی": "Midjourney", "پرپلکسیتی": "Perplexity",
    "انویدیا": "Nvidia", "اِنویدیا": "Nvidia",
    "کلاود کد": "Claude Code", "جِمّا": "Gemma", "جما": "Gemma",
}

# Persian letters + ZWNJ. Used as a word boundary so a key can never be
# replaced mid-word: without this, "پرو" → "Pro" turns "پرونده" (case file)
# into "Proنده". That corruption reached a live preview post.
#
# The class must contain LETTERS ONLY. It used to be the whole U+0600–U+06FF
# block, which also holds Persian punctuation (، ؛ ؟ ٪) and Persian digits
# (۰–۹). A brand followed by a comma therefore failed the boundary test and was
# never repaired: an audited bulletin shipped "مدیرعامل انویدیا، و جرج کورتز"
# while the identical phrase followed by a space became "Nvidia" correctly.
# Persian prose puts a comma after a name constantly, so this single character
# class silently disabled the repair on a large share of real sentences.
_FA_CHAR = (r"\u0620-\u065F"      # Arabic/Persian letters + attached harakat
            r"\u066E-\u06D3"      # extended letters (گ چ پ ژ ک ی …)
            r"\u06D5-\u06DC"      # heh with hamza + attached marks
            r"\u06E5\u06E6\u06EE\u06EF\u06FA-\u06FF"
            r"\u200c")            # ZWNJ (nim-fasele) joins parts of one word
_TRANSLIT_RE = re.compile(
    r"(?<![" + _FA_CHAR + r"])(" + "|".join(
        re.escape(k) for k in sorted(TRANSLIT_FIX, key=len, reverse=True)
    ) + r")(?![" + _FA_CHAR + r"])"
)

# Product-line suffixes are ordinary Persian words on their own (فلش = flash of
# light, پرو = pro/professional, مینی = mini), so they are only converted when
# they trail a Latin product token: "Gemini ۳.۸ فلش" → "Gemini 3.8 Flash",
# while "حافظه فلش" and "پرونده" stay untouched.
_SUFFIX_FIX = {"فلش": "Flash", "توربو": "Turbo", "پرو": "Pro",
               "مینی": "mini", "سایبر": "Cyber", "سونت": "Sonnet",
               "اوپوس": "Opus", "هایکو": "Haiku"}
_SUFFIX_RE = re.compile(
    r"(?P<lead>[A-Za-z][\w\-\.]*(?:[ \u00a0][\d\u06F0-\u06F9][\w\.\u06F0-\u06F9]*)?)"
    r"(?P<sep>[ \u00a0])(?P<word>" + "|".join(_SUFFIX_FIX) + r")"
    r"(?![" + _FA_CHAR + r"])"
)

# Persian digits that sit inside a Latin product name must go back to ASCII so
# the title matches the body: "Gemini ۳.۸ Flash" → "Gemini 3.8 Flash".
_FA_TO_ASCII = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")
_VERSION_IN_LATIN = re.compile(
    r"(?<=[A-Za-z])([ \u00a0\-])([\u06F0-\u06F9]+(?:[.\u066b][\u06F0-\u06F9]+)*)"
)


def _repair_translit(text: str) -> str:
    """Undo Persian transliteration of brand/model names, keeping prose Persian."""
    if not text:
        return text
    text = _TRANSLIT_RE.sub(lambda m: TRANSLIT_FIX[m.group(1)], text)
    text = _SUFFIX_RE.sub(
        lambda m: f"{m.group('lead')}{m.group('sep')}{_SUFFIX_FIX[m.group('word')]}", text)
    # run twice: "Gemini ۳.۸ فلش سایبر" needs two passes to reach سایبر
    text = _SUFFIX_RE.sub(
        lambda m: f"{m.group('lead')}{m.group('sep')}{_SUFFIX_FIX[m.group('word')]}", text)
    text = _VERSION_IN_LATIN.sub(
        lambda m: m.group(1) + m.group(2).translate(_FA_TO_ASCII).replace("\u066b", "."), text)
    return text


# ── spelled-out version numbers ───────────────────────────────────────────
# A live bulletin shipped «نسخه صفر.سی‌وسهار» for version 0.34: the model wrote
# the number in *words*. Nothing caught it — the text is 100% Persian letters so
# the audit passed, and _fa_digits only maps digit→digit. Version numbers must
# always be digits, so spelled-out ones are rewritten from the English source
# title, and if that title has no version to copy the story is rejected rather
# than published with an invented number.
_NUM_WORDS = [
    "صفر", "یک", "دو", "سه", "چهار", "پنج", "شش", "شیش", "هفت", "هشت", "نه",
    "ده", "یازده", "دوازده", "سیزده", "چهارده", "پانزده", "پونزده", "شانزده",
    "شونزده", "هفده", "هیفده", "هجده", "هیجده", "نوزده", "بیست", "سی", "چهل",
    "پنجاه", "شصت", "هفتاد", "هشتاد", "نود", "صد", "دویست", "سیصد", "چهارصد",
    "پانصد", "ششصد", "هفتصد", "هشتصد", "نهصد", "هزار",
]
_NUM_WORD_RE = "(?:" + "|".join(sorted(_NUM_WORDS, key=len, reverse=True)) + ")"
# a chain such as "صفر.سی‌وسه" or "سه و هشت" or "چهار نقطه دو"
_SPELLED_VERSION_RE = re.compile(
    r"(?P<label>نسخه|ورژن|version)\s+"
    r"(?P<num>" + _NUM_WORD_RE + r"(?:[\s\u200c.،/]*(?:و|نقطه|ممیز)?[\s\u200c.]*" + _NUM_WORD_RE + r")*)"
)
_VERSION_IN_SOURCE = re.compile(r"\b[vV]?(\d+(?:\.\d+){0,2})\b")


def _source_version(title_en: str, body: str = "") -> str:
    """The first version-looking number in the English source, or ''."""
    for hay in (title_en, body):
        m = _VERSION_IN_SOURCE.search(hay or "")
        if m:
            return m.group(1)
    return ""


def _fix_spelled_version(text: str, version: str) -> tuple[str, bool]:
    """Replace spelled-out version words with digits.

    Returns (text, ok). ok=False means a spelled version was found but there is
    no real number to substitute, so the caller must not publish the text.
    """
    if not text or "نسخه" not in text and "ورژن" not in text:
        return text, True
    hit = _SPELLED_VERSION_RE.search(text)
    if not hit:
        return text, True
    if not version:
        return text, False
    digits = version.translate(_DIGITS)
    # The model's spelling is often mangled ("سی‌وسهار" for 34), so the regex can
    # stop mid-word and leave a stray fragment ("نسخه ۰.۳۴ار"). Extend the match
    # to the end of the glued word — no space means it is part of the number.
    end = hit.end()
    while end < len(text) and re.match(r"[\u0600-\u06FF\u200c]", text[end]):
        end += 1
    return text[:hit.start()] + f"{hit.group('label')} {digits}" + text[end:], True



# ── orthographic repair ───────────────────────────────────────────────────
# Models occasionally double a letter in a Persian brand name ("آامازون" for
# آمازون, seen in a live run). The audit cannot catch it: the text is 100%
# Persian with no Latin residue, so it scores perfectly. These are the specific
# misspellings observed, plus a general rule for a doubled alef after آ, which
# is never valid Persian orthography.
_SPELLING_FIX = {
    "آامازون": "آمازون", "آمازونن": "آمازون",
    "گوگگل": "گوگل", "گووگل": "گوگل",
    "مایکروسافتت": "مایکروسافت", "مایککروسافت": "مایکروسافت",
    "هووش": "هوش", "مصنوععی": "مصنوعی", "مصنوعیی": "مصنوعی",
    "کاربرران": "کاربران", "شرکتت": "شرکت", "مدلل": "مدل",
}
_DOUBLED_ALEF = re.compile(r"آا")


def _fix_spelling(text: str) -> str:
    """Repair the orthographic doubles that the Persian audit cannot see."""
    if not text:
        return text
    for bad, good in _SPELLING_FIX.items():
        if bad in text:
            text = text.replace(bad, good)
    # آ is already alef-with-madda; a following bare alef is always a typo.
    return _DOUBLED_ALEF.sub("آ", text)


def _localise_one(story: Story) -> Story | None:
    body = (story.summary_en or "").strip()
    # Vendor blogs (OpenAI, DeepMind) often publish an empty RSS description.
    # Without body text the model can only restate the headline, which is how
    # the preview ended up with a single filler fact ("published by DeepMind").
    # The same fetch also yields the page's og:image, so a story that arrived
    # without art in its feed still gets illustrated for free.
    if len(body) < 220 or not story.image:
        fetched, image = fetch_article_and_image(story.url)
        if len(fetched) > len(body):
            body = fetched
        if image and not story.image:
            story.image = image
    body = (body or story.title_en)[:1800]
    prompt = STORY_PROMPT.format(title=story.title_en, source=story.source, body=body)

    for model, tries in ((config.MODEL_FAST, 2), (config.MODEL_STRONG, 1)):
        for _ in range(tries):
            try:
                data = llm.json_call(prompt, model, system=SYSTEM,
                                     temperature=0.25, max_tokens=1100)
            except llm.LLMError:
                continue
            if not isinstance(data, dict):
                continue
            title_fa = _fix_spelling(_repair_translit(str(data.get("title_fa", "")).strip()))
            summary_fa = _fix_spelling(_repair_translit(str(data.get("summary_fa", "")).strip()))
            why_fa = _fix_spelling(_repair_translit(str(data.get("why_fa", "")).strip()))
            facts = [_fix_spelling(_repair_translit(str(f).strip()))
                     for f in (data.get("facts") or []) if str(f).strip()]

            # A version number written in words ("نسخه صفر.سی‌وسه") is a factual
            # defect, not a style one, so it is repaired from the English source
            # or the attempt is thrown away.
            src_version = _source_version(story.title_en, body)
            title_fa, ok_v1 = _fix_spelled_version(title_fa, src_version)
            summary_fa, ok_v2 = _fix_spelled_version(summary_fa, src_version)
            if not (ok_v1 and ok_v2):
                story.facts = []
                continue
            facts = [_fix_spelled_version(f, src_version)[0] for f in facts]

            # Gate every field the reader will actually see.
            ok_title, why_t = llm.audit(title_fa, min_persian_ratio=0.45, max_latin_words=3)
            ok_sum, why_s = llm.audit(summary_fa, min_persian_ratio=0.6, max_latin_words=6)
            if not (ok_title and ok_sum):
                story.facts = []          # discard partial output before retrying
                continue

            clean_facts = []
            for f in facts[:4]:
                ok_f, _ = llm.audit(f, min_persian_ratio=0.45, max_latin_words=3)
                if ok_f and len(f) > 12:
                    clean_facts.append(_fa_digits(f))
            story.title_fa = _fa_digits(title_fa)
            story.summary_fa = _fa_digits(summary_fa)
            story.why_fa = _fa_digits(why_fa) if llm.audit(why_fa, min_persian_ratio=0.5)[0] else ""
            story.facts = clean_facts
            return story
    return None                            # never publish unaudited text


_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")

# A run of digits/dots that is glued to Latin letters belongs to a product name
# (GPT-5.2, cs.AI, Llama-3.1, Paint.net 5.2) and must stay in ASCII. Matching
# the whole run at once is what makes this correct — a per-character check sees
# the "2" in "5.2" as isolated (its neighbours are "." and " ") and converts it,
# producing the mixed-script "GPT-5.۲".
_VERSION_RUN = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:[-_.][A-Za-z0-9]+)*")


def _fa_digits(text: str) -> str:
    """Persian digits, but leave version/product tokens in ASCII."""
    protected: list[tuple[int, int]] = []
    for m in _VERSION_RUN.finditer(text):
        # extend over a trailing " 5.2" style version suffix
        end = m.end()
        tail = re.match(r"[-\s]?\d+(?:\.\d+)*", text[end:])
        if tail:
            end += tail.end()
        protected.append((m.start(), end))

    out = []
    for i, ch in enumerate(text):
        if ch.isdigit() and any(a <= i < b for a, b in protected):
            out.append(ch)
        else:
            out.append(ch.translate(_DIGITS))
    return "".join(out)


def localise(stories: list[Story], workers: int = 4) -> list[Story]:
    with ThreadPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_localise_one, stories))
    ready = [s for s in results if s is not None]
    verify_images(ready, workers=workers)
    return ready


# Telegram fetches a photo URL server-side, and a single unfetchable one fails
# the WHOLE sendRichMessage call ("Bad Request: failed to get HTTP URL content")
# — one dead CDN link would silence the entire bulletin, which is precisely how
# a channel dies quietly. Measured on live feeds: 1 of 12 real image URLs was
# unusable (a Google CMS link served application/octet-stream, and Wired's
# media host answered 500 for a URL that had worked hours earlier).
_MAX_IMAGE_BYTES = 9 * 1024 * 1024   # Telegram's own limit for photo-by-URL


def _image_is_usable(url: str) -> bool:
    """Fetch enough of the URL to prove Telegram will accept it."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
        with urllib.request.urlopen(req, timeout=12) as resp:
            if resp.status != 200:
                return False
            ctype = (resp.headers.get("Content-Type") or "").split(";")[0].strip()
            if not ctype.startswith("image/"):
                return False
            length = int(resp.headers.get("Content-Length") or 0)
            if length and length > _MAX_IMAGE_BYTES:
                return False
            return bool(resp.read(1))
    except Exception:
        return False


def verify_images(stories: list[Story], workers: int = 4) -> None:
    """Drop any image URL that Telegram would choke on. Mutates in place."""
    targets = [s for s in stories if s.image]
    if not targets:
        return
    with ThreadPoolExecutor(max_workers=workers) as ex:
        verdicts = list(ex.map(lambda s: _image_is_usable(s.image), targets))
    for story, ok in zip(targets, verdicts):
        if not ok:
            story.image = ""


DIGEST_PROMPT = """این تیترهای فارسی بولتن امروز است. یک جمع‌بندی فارسی در دو جمله بنویس
که بگوید مهم‌ترین جهت‌گیری این دوره چه بوده. بدون تعارف و بدون فهرست‌کردن دوباره تیترها.
فقط متن جمع‌بندی را برگردان.

{titles}"""


def digest(stories: list[Story]) -> str:
    if not stories:
        return ""
    titles = "\n".join(f"- {s.title_fa}" for s in stories)
    try:
        text = llm.call(DIGEST_PROMPT.format(titles=titles), config.MODEL_FAST,
                        system=SYSTEM, temperature=0.3, max_tokens=400).strip()
    except llm.LLMError:
        return ""
    ok, _ = llm.audit(text, min_persian_ratio=0.6, max_latin_words=6)
    return _fa_digits(text) if ok else ""
