"""Substance test for «نکات کلیدی» — the key-points list under each story.

The user's complaint, made twice: «تو نکات کلیدی فقط یه چندتا جمله از تو خود
خبر انتخاب میشه و گذاشته میشه که اصلا ارزشی نداره». The existing guard only
dropped points that *overlapped* the summary. Auditing live channel post 114
showed why that misses the real problem: all 18 shipped points had 0.00 shingle
overlap with their summaries and were still vacuous —

    «این نسخه عملکرد سایبری گوگل را تقویت می‌کند.»
    «گروه‌های تروریستی در حال حاضر از هوش مصنوعی استفاده می‌کنند.»

Novel wording, zero information. So the test here is positive, not negative: a
point must carry at least one of five kinds of substance, matching the five
categories the prompt already asks for. Measured against those 18 live points:
8 pass, 10 are dropped as filler.

Kinds are also returned so the renderer can label each point by what it is,
which stops the list reading as one undifferentiated block.
"""
from __future__ import annotations

import re

_NUM = re.compile(r"[۰-۹0-9]")

# Comparison against a rival, a previous version, or a ranking.
_COMPARE = ("بیشتر", "کمتر", "بالاتر", "پایین‌تر", "برابر", "نسبت به", "در مقابل",
            "رتبه", "پیشی", "رقیب", "بهتر", "بدتر", "سریع‌تر", "کندتر",
            "دو برابر", "نیمی", "درصد", "٪", "برخلاف", "جای", "سبقت", "همتا")

# A limit, a risk, or a caveat the story itself raises.
_RISK = ("محدودیت", "ریسک", "خطر", "نقص", "هشدار", "ممنوع", "شکست", "نگرانی",
         "ابهام", "چالش", "مشکل", "انتقاد", "تحریم", "شکاف", "سوءاستفاده",
         "آسیب", "افت", "تأخیر", "بحران", "شکایت", "نقض")

# A timeframe or the next concrete step.
_TIME = ("سال", "ماه", "هفته", "روز", "تا پایان", "از سال", "مرحله بعد",
         "زمان‌بندی", "به‌زودی", "تاکنون", "از زمان", "پیش‌نمایش", "بتا",
         "نسخه بعدی", "فصل", "سه‌ماهه", "عرضه می‌شود", "راه‌اندازی می‌شود")

# A magnitude: money, users, parameters, compute.
_SCALE = ("میلیون", "میلیارد", "هزار", "دلار", "یورو", "توکن", "پارامتر",
          "گیگا", "ترا", "پتا", "کاربر", "توسعه‌دهنده", "مگاوات", "وات",
          "بایت", "هسته", "تراشه")

_KINDS = (("عدد", _NUM), ("مقایسه", _COMPARE), ("ریسک", _RISK),
          ("زمان", _TIME), ("مقیاس", _SCALE))

# The salvage prompt lists the five categories by name, and the model sometimes
# echoes the category as a prefix: a live dry run produced
# «محدودیت یا ریسک: بسیاری از ارائه‌دهندگان…». Strip it rather than reject the
# point — the content after the colon is usually good.
_LABEL_PREFIX = re.compile(
    r"^\s*(?:نکته\s*\d*|عدد(?:\s*مشخص)?|مقایسه|محدودیت(?:\s*یا\s*ریسک)?|ریسک"
    r"|بازه\s*زمانی|زمان|مقیاس|مرحله\s*بعدی)\s*[:：\-–—]\s*")

# Vague constructions that sneak past a keyword match. «نسبت به قبل تقویت شده»
# hits the comparison list while saying nothing measurable, so a point built
# only on one of these needs a number to survive.
_VAGUE = ("تقویت شده", "بهبود یافته", "ارتقا یافته", "بهتر شده", "افزایش یافته است",
          "قابلیت‌های عملکردی", "عملکرد بهتری", "پیشرفت مهم", "گام مهم",
          "تحول بزرگ", "آینده روشن", "توجه گسترده", "استقبال خوب")


def strip_label(fact: str) -> str:
    """Remove a leaked category prefix from a key point."""
    return _LABEL_PREFIX.sub("", fact or "").strip()


def kinds_of(fact: str) -> list[str]:
    """Which kinds of substance this point carries. Empty list means filler."""
    if not fact:
        return []
    found = []
    for name, probe in _KINDS:
        if isinstance(probe, re.Pattern):
            if probe.search(fact):
                found.append(name)
        elif any(word in fact for word in probe):
            found.append(name)
    return found


def has_substance(fact: str) -> bool:
    """True when the point says something a reader could act on or verify.

    A vague growth verb ("تقویت شده", "بهبود یافته") matches the comparison
    keywords while carrying no information, so such a point must also bring a
    number. Live dry run caught «قابلیت‌های عملکردی در حوزه سایبری نسبت به قبل
    تقویت شده است» passing on the keyword «نسبت به» alone.
    """
    if not fact:
        return False
    found = kinds_of(fact)
    if not found:
        return False
    if any(v in fact for v in _VAGUE) and not _NUM.search(fact):
        return False
    return True


def primary_kind(fact: str) -> str:
    """The single most informative label for this point, or "" for filler.

    Ordered by what a reader notices first: a hard number beats a magnitude,
    a stated risk beats a timeframe.
    """
    found = kinds_of(fact)
    for name in ("عدد", "مقایسه", "ریسک", "مقیاس", "زمان"):
        if name in found:
            return name
    return ""
