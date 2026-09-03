"""Turn raw English stories into audited Persian editorial content."""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher

from . import config, geo, llm
from . import facts as facts_mod
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
  "impact_fa":  یک جمله: اگر این خبر درست باشد، چه چیزی عملاً تغییر می‌کند؟ برای چه کسی؟
  "facts":      آرایه‌ای از ۲ تا ۴ «نکته کلیدی» فارسی (هر کدام حداکثر ۱۴ کلمه)
  "metric_label": برچسب فارسی مهم‌ترین عدد خبر (مثلاً «ارزش‌گذاری» یا «دقت در MMLU») — اگر عددی نیست، رشته خالی
  "metric_value": همان عدد با ارقام فارسی و واحدش (مثلاً «۱۲ میلیارد دلار» یا «۹۴.۲٪») — اگر نیست، رشته خالی
  "latex":        اگر خبر پژوهشی است و یک رابطه/معیار ریاضی مرکزی دارد، آن را در نحو LaTeX بنویس (مثلاً "\\\\text{{FLOPs}} \\\\approx 6ND")؛ در غیر این صورت رشته خالی

قاعده مهم درباره "facts" — این بخش قبلاً بی‌ارزش بود چون فقط چند جمله از خود خبر کپی می‌شد:
- هیچ نکته‌ای نباید بازنویسی یا کپی جمله‌های "summary_fa" باشد. اگر خواننده خلاصه را خوانده، هر نکته باید چیز *تازه‌ای* به او بدهد.
- هر نکته باید یکی از این‌ها باشد: یک عدد مشخص و معنایش، مقایسه با رقیب یا نسخه قبلی، محدودیت/ریسکی که خبر به آن اشاره می‌کند، بازه زمانی و مرحله بعدی، یا اینکه چه کسی برنده و چه کسی بازنده است.
- ممنوع: جمله‌های کلی مثل «این یک پیشرفت مهم است» یا «کارشناسان می‌گویند آینده روشن است».

قواعد قطعی:
- تمام متن باید فارسی باشد. هیچ کلمه انگلیسی ترجمه‌نشده باقی نماند.
- تنها استثنا: نام شرکت‌ها، مدل‌ها و معیارها به لاتین بماند (OpenAI, GPT-5, Gemini, MMLU).
- نام‌های خاص را با حروف فارسی نویسه‌گردانی نکن. «OpenAI» بنویس، نه «اوپن‌ای‌آی».
- اعداد را با ارقام فارسی بنویس: ۹۴.۲ درصد
- هیچ ادعایی که در متن اصلی نیست اضافه نکن. اگر متن ناقص است، فقط از عنوان استفاده کن و کلیدهای اضافی را خالی بگذار.
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
    return _spread(ranked, keep)


def _spread(ranked: list[Story], keep: int) -> list[Story]:
    """Pick `keep` stories without letting one section or one source swallow it.

    Extracted from `triage` so the balance rules are testable without an LLM
    call — the unbounded-backfill bug that produced post 110 was invisible to
    every existing test because they all stopped at the primary cap.

    Two caps, because section balance alone was not enough. Measured on the
    live 24h pool: 143 stories, 120 of them arXiv across three feeds. arXiv
    papers get classified into every section, so post 112 came out balanced by
    section and still shipped nine research abstracts and ZERO photos — arXiv
    has no article art. Grouping by `source_fa` treats the three arXiv feeds as
    one family, which forces the rest of the bulletin to come from sources that
    do carry pictures.
    """
    picked: list[Story] = []
    per: dict[str, int] = {}
    fam: dict[str, int] = {}

    def family(s: Story) -> str:
        return s.source_fa or s.source

    def take(s: Story) -> None:
        picked.append(s)
        per[s.section] = per.get(s.section, 0) + 1
        fam[family(s)] = fam.get(family(s), 0) + 1

    for s in ranked:
        if per.get(s.section, 0) >= config.MAX_PER_SECTION:
            continue
        if fam.get(family(s), 0) >= config.MAX_PER_FAMILY:
            continue
        take(s)
        if len(picked) >= keep:
            break

    # Backfill when sections were sparse — but the backfill USED TO IGNORE the
    # cap entirely, which is how channel post 110 shipped 6 `models` items out
    # of 9 and eight near-identical arXiv cards. A widened window is mostly
    # arXiv, so the unbounded backfill turned every thin day into a research
    # digest. The ceilings are deliberately looser than the primary caps:
    # filling the bulletin still matters more than perfect balance.
    sec_ceiling = config.MAX_PER_SECTION + config.BACKFILL_SLACK
    fam_ceiling = config.MAX_PER_FAMILY + config.BACKFILL_SLACK
    for s in ranked:
        if len(picked) >= keep:
            break
        if s in picked:
            continue
        if per.get(s.section, 0) >= sec_ceiling:
            continue
        if fam.get(family(s), 0) >= fam_ceiling:
            continue
        take(s)

    # Last resort: a one-sided pool should yield a SHORTER bulletin, not a
    # research digest. Filling all nine slots from the only family left is
    # exactly what produced posts 110 and 112, so this pass stops at
    # MIN_STORIES — enough that the post never looks empty, few enough that it
    # cannot become nine abstracts again.
    floor = min(keep, config.MIN_STORIES)
    for s in ranked:
        if len(picked) >= floor:
            break
        if s not in picked:
            take(s)
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
    # Contracted spellings score 0.80 against the full skeleton — just under the
    # curated floor — so they are listed exactly rather than loosening the floor
    # for every brand. Seen live: «زیرساخت‌های هاگ‌فیس».
    "هاگ‌فیس": "Hugging Face", "هاگفیس": "Hugging Face",
    "هاک‌فیس": "Hugging Face", "هاکفیس": "Hugging Face",
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


def _repair_translit(text: str, source: str = "") -> str:
    """Undo Persian transliteration of brand/model names, keeping prose Persian.

    ``source`` is the English title+body. When given, a second, *fuzzy* pass runs
    for brands the source actually names — see ``_repair_brands_grounded``.
    """
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



# ── source-grounded brand repair ─────────────────────────────────────────
# TRANSLIT_FIX above is a fixed list, so it only ever fixes brands someone
# thought of in advance. A live bulletin shipped «هاکینگ فیس» (the dictionary
# holds «هاگینگ فیس», with گ) and another shipped «کراداستریک» for CrowdStrike.
# Both are invisible to the Persian audit: the text is 100% Persian letters.
#
# This pass is *grounded*: it only ever substitutes a name that literally
# appears in the English source of that same story, matched by consonant
# skeleton, so it cannot invent a brand or rewrite ordinary Persian prose.
_LAT_SKEL = [("sch", "S"), ("sh", "Š"), ("ch", "Č"), ("ph", "F"), ("th", "T"),
             ("ck", "K"), ("qu", "KV"), ("gh", "G"), ("kh", "X"), ("wh", "V"),
             # Soft c: "Face" is /feɪs/, so its skeleton must be FS, not FK.
             # Without this, «فیس» scored 0.67 against "Hugging Face" and the
             # real live defect «هاکینگ فیس» was never repaired.
             ("ce", "S"), ("ci", "S"), ("cy", "S")]
_LAT_MAP = {"b": "B", "p": "P", "f": "F", "v": "V", "w": "V", "k": "K",
            "c": "K", "q": "K", "g": "K", "d": "D", "t": "T", "s": "S",
            "z": "S", "x": "KS", "j": "J", "m": "M", "n": "N", "r": "R",
            "l": "L", "h": "H", "y": "", "a": "", "e": "", "i": "", "o": "",
            "u": ""}
# ک and گ collapse to one symbol, as do g and k above: Persian writers swap the
# voiced/voiceless velar freely when transliterating (the live defect wrote
# «هاکینگ» for "Hugging"). Merging them is what makes that pair reachable.
_FA_MAP = {"ب": "B", "پ": "P", "ف": "F", "و": "V", "ک": "K", "ق": "K",
           "گ": "K", "غ": "K", "د": "D", "ذ": "S", "ت": "T", "ط": "T",
           "س": "S", "ص": "S", "ث": "S", "ز": "S", "ض": "S", "ظ": "S",
           "ش": "Š", "چ": "Č", "ج": "J", "ژ": "J", "م": "M", "ن": "N",
           "ر": "R", "ل": "L", "ه": "H", "ح": "H", "خ": "X", "ی": "",
           "ا": "", "آ": "", "أ": "", "إ": "", "ع": "", "ء": "", "ؤ": "",
           "\u200c": "", "\u0654": ""}
# Persian words that merely *look* like a brand skeleton. Substituting inside
# these would corrupt real prose, exactly like the «پرونده» → «Proنده» bug.
_NOT_A_BRAND = {
    "این", "آن", "برای", "است", "شده", "شرکت", "مدل", "هوش", "مصنوعی", "کاربران",
    "سیستم", "داده", "داده‌ها", "پلتفرم", "نسخه", "جدید", "خود", "درصد", "دلار",
    "میلیون", "میلیارد", "هزار", "سال", "ماه", "روز", "بر", "با", "از", "به",
    "که", "را", "در", "و", "یا", "تا", "اما", "اگر", "هم", "نیز", "بیش", "کمتر",
    "توسط", "طور", "همچنین", "براساس", "بر‌اساس", "گزارش", "خبر", "منبع",
    "تصویر", "ویدیو", "متن", "کد", "ابزار", "قابلیت", "امکان", "افزایش",
    "کاهش", "توانایی", "پژوهش", "تحقیق", "مقاله", "پردازنده", "گرافیکی",
    "حافظه", "شبکه", "زبانی", "بزرگ", "کوچک", "سریع", "دقت", "هزینه",
}
# Brands Persian journalism routinely writes in Persian letters: leave alone.
_KEEP_PERSIAN = {"گوگل", "مایکروسافت", "آمازون", "اپل", "متا", "تسلا", "سامسونگ",
                 "اینتل", "توییتر", "فیسبوک", "ادوبی", "اوراکل", "هوآوی",
                 "شیائومی", "سونی", "ال‌جی", "بایدو", "علی‌بابا", "تنسنت"}
_STOP_EN = {"The", "This", "That", "These", "Those", "A", "An", "And", "But",
            "For", "With", "From", "Its", "It", "In", "On", "At", "By", "To",
            "As", "Of", "New", "Now", "How", "Why", "What", "When", "Where",
            "We", "You", "They", "He", "She", "I", "If", "Is", "Are", "Was",
            "Were", "Be", "Been", "Has", "Have", "Had", "Will", "Can", "May",
            "More", "Most", "Best", "First", "Last", "Two", "One", "Three",
            "AI", "API", "CEO", "CTO", "US", "UK", "EU", "But", "So", "Not"}
# Candidate capitalisation. Headline case capitalises every word, so "Safety",
# "Awareness", "Benchmark" and "Garrett" all look like brands — a measured live
# run turned «سال جاری» into «سال Garrett», «این بنچمارک» into «این Benchmark»
# and «۲۸۸۰ در پیکسل» into «۲۸۸۰ در PCs», dropping that bulletin from 98.7% to
# 93.8% Persian. Two shapes are unambiguous anywhere in a sentence:
#   * an internal capital after a lowercase letter — CrowdStrike, OpenAI, NeoMME
#   * an all-caps run of 3+ — NVIDIA, MIT, IBM
# Everything else (Mistral, Hugging Face) needs the extra filters below.
_BRAND_STRONG = re.compile(r"^(?:[A-Za-z]*[a-z][A-Z][A-Za-z0-9]*|[A-Z]{3,}[a-z]?)$")
_BRAND_EN = re.compile(r"\b([A-Z][a-zA-Z0-9]*(?:[ \u00a0][A-Z][a-zA-Z0-9]+){0,2})\b")

# A capitalised word that is also ordinary English vocabulary is a *word*, not a
# brand: rendering it in Persian is translation, which is what the channel is
# for. Only names outside this list may be pulled back into Latin. (No system
# word list is installed — checked /usr/share/dict — so the common cases that
# actually appear in AI headlines are enumerated.)
_COMMON_EN = {
    "safety", "awareness", "benchmark", "benchmarks", "security", "privacy",
    "research", "researchers", "model", "models", "system", "systems", "data",
    "agent", "agents", "tool", "tools", "chip", "chips", "cloud", "code",
    "compute", "computing", "content", "context", "control", "cost", "costs",
    "court", "deal", "design", "developer", "developers", "device", "devices",
    "energy", "enterprise", "evaluation", "framework", "future", "growth",
    "health", "image", "images", "impact", "industry", "inference", "insight",
    "intelligence", "investment", "language", "launch", "law", "laws", "layer",
    "learning", "level", "license", "market", "memory", "mind", "money",
    "network", "news", "note", "notes", "open", "order", "paper", "papers",
    "people", "performance", "phone", "phones", "platform", "policy", "power",
    "price", "prices", "product", "products", "program", "project", "prompt",
    "release", "report", "reports", "review", "reviews", "risk", "rules",
    "scale", "science", "search", "series", "service", "software", "source",
    "speech", "speed", "standard", "state", "story", "study", "support",
    "team", "tech", "test", "tests", "text", "time", "today", "training",
    "trust", "update", "user", "users", "value", "version", "video", "vision",
    "voice", "web", "week", "work", "world", "year", "years", "police",
    "camera", "cameras", "school", "schools", "city", "state", "county",
    "hugging",   # only meaningful as part of "Hugging Face"
}


def _skeleton_latin(word: str) -> str:
    low = word.lower()
    for pair, rep in _LAT_SKEL:
        low = low.replace(pair, rep.lower() if rep.isupper() else rep)
    out = []
    for ch in low:
        out.append(_LAT_MAP.get(ch, "" if not ch.isalnum() else ch.upper()))
    skel = "".join(out)
    return re.sub(r"(.)\1+", r"\1", skel)      # "gg" and "tt" are one sound


def _skeleton_fa(word: str) -> str:
    out = "".join(_FA_MAP.get(ch, "") for ch in word)
    return re.sub(r"(.)\1+", r"\1", out)


def _latin_brands(source: str) -> list[str]:
    """Names in the English source that may be restored, longest first.

    Three filters, each added because of a measured live defect:
      * edge stop-words are trimmed, and multi-word candidates also contribute
        their parts — headline case glues a job title on ("CrowdStrike CEO"),
        which produced «مدیرعامل CrowdStrike CEO» in a real run;
      * a single word must either look like a brand (internal capital or an
        all-caps run) or be absent from ordinary English vocabulary, so
        "Safety"/"Awareness"/"Benchmark" no longer qualify;
      * a multi-word phrase is kept only if at least one part qualifies, which
        keeps "Hugging Face" and "Flock Safety" while dropping "Police Camera".
    """
    found = []

    def qualifies(word: str) -> bool:
        return bool(_BRAND_STRONG.match(word)) or word.lower() not in _COMMON_EN

    def add(name: str) -> None:
        if len(name) < 3 or name in found:
            return
        parts = name.split()
        if len(parts) == 1 and (parts[0] in _STOP_EN or not qualifies(parts[0])):
            return
        if not any(qualifies(p) for p in parts):
            return
        found.append(name)

    for m in _BRAND_EN.finditer(source or ""):
        parts = m.group(1).strip().split()
        while parts and parts[0] in _STOP_EN:
            parts.pop(0)
        while parts and parts[-1] in _STOP_EN:
            parts.pop()
        if not parts:
            continue
        add(" ".join(parts))
        if len(parts) > 1:
            for p in parts:
                add(p)
    return sorted(found, key=len, reverse=True)


_FA_WORD = re.compile(r"[\u0620-\u06FF\u200c]{3,}")

# Minimum consonant-skeleton length. "Garrett" reduces to KRT, which scores 1.00
# against «کارت» (card) and 0.86 against «کارت‌های»; live message 5061 shipped
# «تا پایان سال جاری میلادی Garrett اجباری» because of it. Every real win has a
# longer skeleton (Hugging Face HKNKFS, CrowdStrike KRVDSTRK, Mistral MSTRL).
_MIN_SKEL = 5
# Ratio floors. The curated list is a closed set of known AI brands, so a looser
# floor is safe there; source-derived names are open-ended, and 0.83 let
# «مقیاس بزرگ‌تر» (BSRKTR) match "Abstract" (BSTRKT) in live message 5063.
_CURATED_MIN_RATIO = 0.82
_GROUNDED_MIN_RATIO = 0.92
# Brands worth matching fuzzily: the values of TRANSLIT_FIX are the canonical
# spellings, so variants of a *known* brand are caught even when the exact
# Persian string is absent from the dictionary — «هاکینگ فیس» vs «هاگینگ فیس».
_CURATED_SKELS = sorted(
    {v for v in TRANSLIT_FIX.values()},
    key=len, reverse=True,
)
_CURATED_SKELS = [(b, _skeleton_latin(b), _CURATED_MIN_RATIO) for b in _CURATED_SKELS]
_CURATED_SKELS = [t for t in _CURATED_SKELS if len(t[1]) >= _MIN_SKEL]


def _repair_brands_grounded(text: str, source: str) -> str:
    """Restore Latin brand names that the model spelled in Persian letters.

    Two passes, deliberately narrow, because the risk is asymmetric: a missed
    repair leaves a Persian spelling (which is what the channel wants anyway),
    while a false one injects an English word into Persian prose — the user's
    single biggest complaint.

      * curated pass — fuzzy-matches against the brand names already listed in
        TRANSLIT_FIX, so spelling variants of a *known* brand are caught. The
        live defect «هاکینگ فیس» is exactly this: the dictionary holds «هاگینگ
        فیس» with گ, and one letter defeated the exact-string lookup.
      * grounded pass — names taken from this story's own English text, limited
        to shapes that can only be brands (internal capital, or an all-caps run
        of 3+). Headline case capitalises every word, so without that limit
        "Abstract", "Larger", "Garrett", "Safety" and "Benchmark" all became
        substitution candidates and corrupted real sentences in live runs
        («مقیاس بزرگ‌تر» → «مقیاس Abstract», «سال جاری» → «سال Garrett»).
    """
    if not text:
        return text
    skels = list(_CURATED_SKELS)
    for brand in _latin_brands(source):
        skel = _skeleton_latin(brand)
        if len(skel) >= _MIN_SKEL:
            skels.append((brand, skel, _GROUNDED_MIN_RATIO))
    if not skels:
        return text

    words = [(m.start(), m.end(), m.group(0)) for m in _FA_WORD.finditer(text)]
    replacements = []          # (start, end, latin)
    i = 0
    while i < len(words):
        matched = False
        # Try the longest phrase first: "هاکینگ فیس" must beat "هاکینگ".
        for span in (3, 2, 1):
            if i + span > len(words):
                continue
            chunk = words[i:i + span]
            phrase = text[chunk[0][0]:chunk[-1][1]]
            if any(w[2] in _KEEP_PERSIAN or w[2] in _NOT_A_BRAND for w in chunk):
                continue
            fa_skel = _skeleton_fa(phrase.replace(" ", ""))
            if len(fa_skel) < _MIN_SKEL:
                continue
            best, ratio, need = "", 0.0, 1.0
            for brand, skel, floor in skels:
                if abs(len(skel) - len(fa_skel)) > 2:
                    continue
                r = SequenceMatcher(None, skel, fa_skel).ratio()
                if r > ratio:
                    best, ratio, need = brand, r, floor
            if best and ratio >= need and len(best.split()) >= len(phrase.split()) - 1:
                replacements.append((chunk[0][0], chunk[-1][1], best))
                i += span
                matched = True
                break
        if not matched:
            i += 1

    for start, end, latin in reversed(replacements):
        text = text[:start] + latin + text[end:]
    return text


# ── English words the model leaves untranslated ───────────────────────────
# The audit counts Latin *words*, so a couple of stragglers per sentence always
# pass it. Measured on live message 5065: «ماه September» and «فناوری Rendering
# عصبی» — ordinary vocabulary, not brands, and precisely the "mostly Persian
# with English sprinkled in" outcome the user rejected. Brands stay Latin; only
# common nouns and calendar words are forced back to Persian.
_EN_TO_FA = {
    "January": "ژانویه", "February": "فوریه", "March": "مارس", "April": "آپریل",
    "May": "مه", "June": "ژوئن", "July": "جولای", "August": "اوت",
    "September": "سپتامبر", "October": "اکتبر", "November": "نوامبر",
    "December": "دسامبر",
    "Monday": "دوشنبه", "Tuesday": "سه‌شنبه", "Wednesday": "چهارشنبه",
    "Thursday": "پنجشنبه", "Friday": "جمعه", "Saturday": "شنبه", "Sunday": "یکشنبه",
    "Rendering": "رندرینگ", "rendering": "رندرینگ",
    "Benchmark": "بنچمارک", "benchmark": "بنچمارک",
    "Awareness": "آگاهی", "awareness": "آگاهی",
    "Safety": "ایمنی", "Security": "امنیت", "Privacy": "حریم خصوصی",
    "Abstract": "چکیده", "Larger": "بزرگ‌تر", "Police": "پلیس",
    "Camera": "دوربین", "Cameras": "دوربین‌ها",
    "Framework": "فریم‌ورک", "framework": "فریم‌ورک",
    "Prompt": "پرامپت", "Dataset": "مجموعه‌داده", "dataset": "مجموعه‌داده",
    "Inference": "استنتاج", "inference": "استنتاج",
    "Fine-tuning": "ریزتنظیم", "fine-tuning": "ریزتنظیم",
    "Open Source": "متن‌باز", "open source": "متن‌باز",
    "Startup": "استارتاپ", "startup": "استارتاپ",
}
# Only replace a standalone word, and never one glued to a Latin neighbour: in
# "Visual Concepts" or "GeForce NOW" the parts belong to a name.
_EN_TO_FA_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(
        re.escape(k) for k in sorted(_EN_TO_FA, key=len, reverse=True)
    ) + r")(?![A-Za-z0-9])"
)


def _translate_stragglers(text: str) -> str:
    """Force common English words the model left behind into Persian."""
    if not text:
        return text

    def keep_names(m: re.Match) -> str:
        start, end = m.start(1), m.end(1)
        before = text[:start].rstrip()
        after = text[end:].lstrip()
        # A capitalised Latin neighbour means this is part of a proper name.
        if re.search(r"[A-Za-z0-9][ \u00a0]?$", before) or re.match(r"^[A-Z0-9]", after):
            return m.group(1)
        return _EN_TO_FA[m.group(1)]

    return _fuse_hybrids(_EN_TO_FA_RE.sub(keep_names, text))


# Hybrid words are the worst residue because the audit cannot see them: the token
# «خودregressive» (live message 5119, from "autoregressive") is mostly Persian, so
# the Latin-word count barely moves and the ratio stays high. They occur when the
# model translates the prefix and abandons the stem. Fixing the stems that
# actually appear in AI writing is cheap; the general rule (Persian letter glued
# directly to a lowercase Latin run) then catches the rest by stripping the
# fragment rather than shipping it.
_HYBRID_STEM = {
    "regressive": "بازگشتی", "supervised": "نظارت‌شده", "attention": "توجه",
    "encoder": "رمزگذار", "decoder": "رمزگشا", "transformer": "ترنسفورمر",
    "embedding": "برداری", "tuning": "تنظیم", "training": "آموزش",
    "inference": "استنتاج", "learning": "یادگیری", "modal": "وجهی",
    "lingual": "زبانه", "scaling": "مقیاس‌پذیری", "grained": "دانه",
}
_HYBRID = re.compile(r"([\u0600-\u06FF\u200c]+)([a-z]{4,})")


def _fuse_hybrids(text: str) -> str:
    def fix(m: re.Match) -> str:
        head, tail = m.group(1), m.group(2)
        if tail in _HYBRID_STEM:
            return f"{head}{_HYBRID_STEM[tail]}"
        return head          # drop an unrecognised Latin tail, keep the Persian
    return _HYBRID.sub(fix, text)


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

# Letters from the Arabic block that are NOT Persian. The translation model
# reaches for them when it transliterates a brand name — live dry run produced
# «بڈراک» for Bedrock, with U+0688 (Urdu ḍāl). To a Persian reader that is a
# foreign letter in the middle of a word, and the Persian-ratio audit is blind
# to it: the character counts as Arabic-script, so the ratio does not move.
# Mapped to the nearest Persian letter rather than deleted, so a mistrans-
# literation degrades to a readable word instead of a hole.
_FOREIGN_LETTERS = str.maketrans({
    "ك": "ک", "ي": "ی", "ى": "ی", "ﻯ": "ی",   # Arabic kaf/yeh forms
    "ة": "ه", "ﺓ": "ه",
    "أ": "ا", "إ": "ا", "ٱ": "ا", "آ": "آ",
    "ؤ": "و", "ئ": "ئ",
    "ڈ": "د", "ٹ": "ت", "ڑ": "ر", "ں": "ن",   # Urdu retroflex/nasal
    "ھ": "ه", "ے": "ی", "ۓ": "ی",
    "ٰ": "", "ٓ": "", "ٔ": "",                  # stray Arabic diacritics
})


def _normalise_script(text: str) -> str:
    """Fold non-Persian Arabic-script letters onto their Persian equivalents."""
    return text.translate(_FOREIGN_LETTERS) if text else text


def _fix_spelling(text: str) -> str:
    """Repair the orthographic doubles that the Persian audit cannot see."""
    if not text:
        return text
    for bad, good in _SPELLING_FIX.items():
        if bad in text:
            text = text.replace(bad, good)
    text = _normalise_script(text)
    # آ is already alef-with-madda; a following bare alef is always a typo.
    return _DOUBLED_ALEF.sub("آ", text)


FACTS_PROMPT = """از متن این خبر فقط «نکات کلیدی» سختِ قابل‌استناد را بیرون بکش.

عنوان: {title}
متن: {body}

یک شیء JSON با کلید "facts" برگردان: آرایه‌ای از ۲ تا ۴ رشته فارسی.

هر نکته باید *دقیقاً* یکی از این پنج نوع باشد و آن نوع باید در خودِ جمله دیده شود:
۱. عدد مشخص و معنایش (قیمت، دقت، تعداد پارامتر، امتیاز بنچمارک، مبلغ سرمایه)
۲. مقایسه با رقیب یا نسخه قبلی (چه چیزی بهتر/بدتر/سریع‌تر شد و چقدر)
۳. محدودیت یا ریسکی که خودِ خبر به آن اشاره کرده
۴. بازه زمانی یا مرحله بعدی (چه زمانی، تا کی، بعد از این چه)
۵. مقیاس (تعداد کاربر، حجم داده، توان مصرفی، اندازه بازار)

ممنوعِ مطلق — این‌ها را ننویس:
- جمله‌های کلی: «این یک پیشرفت مهم است»، «عملکرد را تقویت می‌کند»، «تمرکز بر ... است»
- بازنویسی عنوان یا خلاصه خبر
- هر چیزی که عدد، مقایسه، ریسک، زمان یا مقیاس در آن نباشد

اگر متن هیچ نکته سختی ندارد، آرایه خالی برگردان. آرایه خالی بهتر از نکته بی‌ارزش است.
اعداد را با ارقام فارسی بنویس. فقط JSON برگردان."""


def _salvage_facts(story: Story, body: str, source: str, clean: callable) -> list[str]:
    """A second, narrower pass when the first call produced only filler.

    Measured on live post 114: of 18 shipped points, 10 carried no number, no
    comparison, no risk, no timeframe and no magnitude. The substance filter
    drops those — which would leave several stories with an empty «نکات کلیدی»
    section. Rather than show nothing, ask again with a prompt that does one job
    only. An empty result is still accepted: no section beats a worthless one.
    """
    try:
        data = llm.json_call(
            FACTS_PROMPT.format(title=story.title_en, body=body[:1800]),
            config.MODEL_STRONG, system=SYSTEM, temperature=0.15, max_tokens=600)
    except llm.LLMError:
        return []
    if isinstance(data, dict):
        raw = data.get("facts") or []
    elif isinstance(data, list):
        raw = data
    else:
        return []

    out: list[str] = []
    for item in raw[:4]:
        f = clean(facts_mod.strip_label(str(item).strip()))
        if len(f) < 13:
            continue
        ok, _ = llm.audit(f, min_persian_ratio=0.45, max_latin_words=3)
        if not ok:
            continue
        if not facts_mod.has_substance(f):
            continue
        if _too_similar(f, story.summary_fa):
            continue
        if any(_too_similar(f, prev) for prev in out):
            continue
        out.append(_fa_digits(f))
    return out


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
            # `source` grounds the fuzzy brand repair: only names that literally
            # appear in this story's own English text can be substituted.
            source = f"{story.title_en}\n{body}"

            def _clean(value: str) -> str:
                return _translate_stragglers(_repair_brands_grounded(
                    _fix_spelling(_repair_translit(str(value).strip())), source))

            title_fa = _clean(data.get("title_fa", ""))
            summary_fa = _clean(data.get("summary_fa", ""))
            why_fa = _clean(data.get("why_fa", ""))
            facts = [_clean(facts_mod.strip_label(f))
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
            for f in facts[:5]:
                ok_f, _ = llm.audit(f, min_persian_ratio=0.45, max_latin_words=3)
                if not (ok_f and len(f) > 12):
                    continue
                # The user's complaint about "نکات کلیدی" was that it held
                # sentences lifted from the story. Prompting alone does not fix
                # that reliably, so near-copies of the summary are dropped here
                # and each point must also differ from its siblings.
                if _too_similar(f, story.summary_fa or summary_fa):
                    continue
                if any(_too_similar(f, prev) for prev in clean_facts):
                    continue
                # Novel wording is not the same as substance. Live post 114
                # shipped 18 points with ZERO summary overlap, 10 of which said
                # nothing ("این نسخه عملکرد سایبری گوگل را تقویت می‌کند"). A
                # point must now carry a number, a comparison, a stated risk, a
                # timeframe, or a magnitude.
                if not facts_mod.has_substance(f):
                    continue
                clean_facts.append(_fa_digits(f))
            story.title_fa = _fa_digits(title_fa)
            story.summary_fa = _fa_digits(summary_fa)
            story.why_fa = _fa_digits(why_fa) if llm.audit(why_fa, min_persian_ratio=0.5)[0] else ""
            story.facts = clean_facts
            # Everything the model offered was filler. Ask once more with a
            # prompt that asks ONLY for hard points, using the stronger model.
            if len(clean_facts) < 2:
                story.facts = _salvage_facts(story, body, source, _clean) or clean_facts

            impact = _clean(data.get("impact_fa", ""))
            if impact and not _too_similar(impact, story.why_fa) \
                    and llm.audit(impact, min_persian_ratio=0.5)[0]:
                story.impact_fa = _fa_digits(impact)

            label = str(data.get("metric_label", "")).strip()
            value = str(data.get("metric_value", "")).strip()
            if label and value and len(label) < 40 and len(value) < 40:
                story.metric_label = _fa_digits(_clean(label))
                story.metric_value = _fa_digits(value)

            # LaTeX is passed through untouched: it is markup, not prose, so the
            # Persian audit and digit conversion must not run over it. It must
            # actually look like an expression — models otherwise return a plain
            # sentence here, which renders as broken output.
            latex = str(data.get("latex", "")).strip()
            if latex and len(latex) < 160 and ("\\" in latex or any(c in latex for c in "^_{}")):
                story.latex = latex
            return story
    return None                            # never publish unaudited text


def _shingles(text: str) -> set[str]:
    words = [w for w in re.findall(r"[^\W_]+", text or "", re.UNICODE) if len(w) > 2]
    return {" ".join(words[i:i + 3]) for i in range(max(0, len(words) - 2))}


def _too_similar(a: str, b: str, threshold: float = 0.4) -> bool:
    """True when `a` mostly repeats `b` (word-trigram overlap).

    Trigrams rather than single words: "OpenAI" appearing in both a summary and
    a key point is fine, a shared run of three words means the sentence was
    copied. Measured against real bulletins, 0.4 rejects lifted sentences while
    leaving genuinely new points (numbers, comparisons, risks) untouched.
    """
    sa, sb = _shingles(a), _shingles(b)
    if not sa or not sb:
        return False
    return len(sa & sb) / len(sa) >= threshold


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
    decorate(ready)
    return ready


def decorate(stories: list[Story]) -> None:
    """Attach the blocks only some stories get, so posts differ in SHAPE.

    A citation card for every arXiv paper (cheap, offline, no network), and a
    map for at most ONE story per bulletin — the first whose English text names
    a place the geocoder recognises. Capping it at one is deliberate: three maps
    in a row would be the same monotony the user complained about, in map form.
    """
    for s in stories:
        s.citation = geo.citation(s.url, s.title_en, s.source)

    geo.reset()
    for s in stories:
        hit = geo.locate(f"{s.title_en}\n{s.summary_en}")
        if hit:
            s.map_lat, s.map_lon, s.map_label = hit
            return


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
    # The digest skipped the prose pipeline entirely, so a stray Urdu/Arabic
    # letter from the model went straight to the top of the bulletin.
    return _fa_digits(_fix_spelling(text)) if ok else ""
