"""Substance test for «نکات کلیدی» — the key-points list under each story.

The user's complaint, made twice: «تو نکات کلیدی فقط یه چندتا جمله از تو خود
خبر انتخاب میشه و گذاشته میشه که اصلا ارزشی نداره». An overlap check against the
summary was not enough — live channel post 114 shipped 18 points with 0.00
shingle overlap that still said nothing:

    «این نسخه عملکرد سایبری گوگل را تقویت می‌کند.»
    «گروه‌های تروریستی در حال حاضر از هوش مصنوعی استفاده می‌کنند.»

So the test here is positive: a point must carry a number, a comparison, a stated
risk, a timeframe, or a magnitude — the five categories the prompt asks for.

Three refinements that live posts forced, each of which had let filler through:

1. **Word boundaries.** Plain `in` matching is wrong for Persian. «دریافت»
   contains «افت» (loss) and «مصنوعی» contains «صنعت»; post 118 shipped
   «گزارش‌ها حاکی از دریافت ایمیل‌های متعدد…» labelled ⚠️ risk purely because of
   that. Keywords must start at a word boundary, with Persian suffixes
   («کاربران», «مدل‌ها») still allowed.
2. **A magnitude needs a magnitude.** «کاربران هوش مصنوعی» matched the scale
   list while quantifying nothing. Scale words now require a digit.
3. **Soft comparatives need a number.** «کنترل بهتر فرآیندها» and «نسبت به قبل
   تقویت شده» are not comparisons a reader can check. Ranking words (رتبه،
   سبقت، در مقابل) still stand alone; bare comparatives do not.

Kinds are also returned so the renderer can label each point by what it is,
which stops the list reading as one undifferentiated block.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"[۰-۹0-9]")

# A digit that belongs to a product/version token ("Apache 2.0", "K2 0.9B") is
# not a fact. Matches the protection in enrich._fa_digits, which leaves exactly
# these runs in ASCII.
_VERSION_TOKEN = re.compile(r"[A-Za-z][A-Za-z.\-]*\s*\d+(?:\.\d+)*[A-Za-z]*")

# Persian letters plus ZWNJ: what may sit next to a keyword without breaking it.
_LETTER = re.compile(r"[\u0600-\u06FF\u200c]")

# Suffixes a keyword may legitimately grow: «کاربران», «مدل‌ها», «سریع‌تری».
_SUFFIX = re.compile(r"(?:ها|های|ان|ی|یی|تر|تری|ترین|شده|شدن)?[\u200c]?")

# ── the five kinds ────────────────────────────────────────────────────────
# Ranking and explicit contrast: informative on their own.
_COMPARE_HARD = ("رتبه", "پیشی", "سبقت", "در مقابل", "برخلاف", "دو برابر",
                 "درصد", "٪", "رکورد")
# Bare comparatives: only meaningful with a number attached.
_COMPARE_SOFT = ("بیشتر", "کمتر", "بالاتر", "پایین‌تر", "برابر", "نسبت به",
                 "بهتر", "بدتر", "سریع‌تر", "کندتر", "نیمی", "همتا", "رقیب")

_RISK = ("محدودیت", "ریسک", "خطر", "نقص", "هشدار", "ممنوع", "شکست", "نگرانی",
         "ابهام", "چالش", "مشکل", "انتقاد", "تحریم", "شکاف", "سوءاستفاده",
         "آسیب", "افت", "تأخیر", "بحران", "شکایت", "نقض", "تهدید")

# «محدودیت نرخ» is rate limiting — a feature name, not a risk. Post 118 shipped
# it as ⚠️. Same for «بدون محدودیت», which is a selling point.
_RISK_FALSE_FRIENDS = ("محدودیت نرخ", "بدون محدودیت", "محدودیت‌های نرخ",
                       "رفع شکاف", "کاهش شکاف", "بدون ریسک", "رفع مشکل",
                       "رفع نقص", "حل چالش")

_TIME = ("سال", "ماه", "هفته", "روز", "تا پایان", "از سال", "مرحله بعد",
         "زمان‌بندی", "به‌زودی", "تاکنون", "از زمان", "پیش‌نمایش", "بتا",
         "نسخه بعدی", "فصل", "سه‌ماهه", "عرضه می‌شود", "راه‌اندازی می‌شود")

_SCALE = ("میلیون", "میلیارد", "هزار", "دلار", "یورو", "توکن", "پارامتر",
          "گیگا", "ترا", "پتا", "کاربر", "توسعه‌دهنده", "مگاوات", "وات",
          "بایت", "هسته", "تراشه", "گیگاوات")

# The salvage prompt names the five categories, and the model sometimes echoes
# one back as a prefix: «محدودیت یا ریسک: بسیاری از ارائه‌دهندگان…» shipped live.
_LABEL_PREFIX = re.compile(
    r"^\s*(?:نکته\s*\d*|عدد(?:\s*مشخص)?|مقایسه|محدودیت(?:\s*یا\s*ریسک)?|ریسک"
    r"|بازه\s*زمانی|زمان|مقیاس|مرحله\s*بعدی)\s*[:：\-–—]\s*")

# Vague constructions that carry a keyword but no information.
_VAGUE = ("تقویت شده", "بهبود یافته", "ارتقا یافته", "بهتر شده", "افزایش یافته است",
          "قابلیت‌های عملکردی", "عملکرد بهتری", "پیشرفت مهم", "گام مهم",
          "تحول بزرگ", "آینده روشن", "توجه گسترده", "استقبال خوب",
          "کمک می‌کند", "تمرکز اصلی")


def strip_label(fact: str) -> str:
    """Remove a leaked category prefix from a key point."""
    return _LABEL_PREFIX.sub("", fact or "").strip()


def _has_word(text: str, word: str) -> bool:
    """True when `word` appears as a word, not buried inside a longer one.

    Persian has no case and few delimiters, so `in` produces false positives
    that are impossible to spot by eye: «افت» inside «دریافت» labelled a live
    key point as a risk. A keyword must therefore begin at a non-letter (or at
    the start of the string) and end at a non-letter, allowing the common
    inflectional suffixes.
    """
    for m in re.finditer(re.escape(word), text):
        start, end = m.start(), m.end()
        if start and _LETTER.match(text[start - 1]):
            continue                          # buried: a letter precedes it
        tail = _SUFFIX.match(text, end)
        end = tail.end() if tail else end
        if end < len(text) and _LETTER.match(text[end]):
            continue                          # buried: real word continues
        return True
    return False


def _any_word(text: str, words) -> bool:
    return any(_has_word(text, w) for w in words)


def _real_number(text: str) -> bool:
    """A digit that is not merely part of a version/product token."""
    stripped = _VERSION_TOKEN.sub(" ", text)
    return bool(_NUM.search(stripped))


def kinds_of(fact: str) -> list[str]:
    """Which kinds of substance this point carries. Empty list means filler."""
    if not fact:
        return []
    numeric = _real_number(fact)
    found: list[str] = []

    if numeric:
        found.append("عدد")

    if _any_word(fact, _COMPARE_HARD) or (numeric and _any_word(fact, _COMPARE_SOFT)):
        found.append("مقایسه")

    if _any_word(fact, _RISK) and not any(ff in fact for ff in _RISK_FALSE_FRIENDS):
        found.append("ریسک")

    if _any_word(fact, _TIME):
        found.append("زمان")

    # A magnitude with no digits quantifies nothing («کاربران هوش مصنوعی»).
    if numeric and _any_word(fact, _SCALE):
        found.append("مقیاس")

    return found


def has_substance(fact: str) -> bool:
    """True when the point says something a reader could act on or verify."""
    if not fact:
        return False
    if not kinds_of(fact):
        return False
    # A vague growth verb with no number is a claim, not a fact.
    if any(v in fact for v in _VAGUE) and not _real_number(fact):
        return False
    return True


def primary_kind(fact: str) -> str:
    """The single most informative label for this point, or "" for filler.

    Ordered by what a reader notices first: a magnitude is a more specific claim
    than a bare number, and a stated risk beats a timeframe.
    """
    found = kinds_of(fact)
    for name in ("مقیاس", "مقایسه", "ریسک", "عدد", "زمان"):
        if name in found:
            return name
    return ""


# Kept for the tests that enumerate every kind the filter can return.
_KINDS = (("عدد", _NUM), ("مقایسه", _COMPARE_HARD), ("ریسک", _RISK),
          ("زمان", _TIME), ("مقیاس", _SCALE))
