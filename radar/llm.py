"""LLM client for the router, plus the Persian translation quality gate.

The old project's failure mode was structural: it built Persian text by
substituting words from a hand-written English→Persian table
(`_generate_persian_title`), so anything not in the table stayed English and
some entries were simply wrong ('GPT' → 'جی‌پی‌یو', which is GPU). The real
translator existed but only ran as a fallback.

Here translation is the ONLY path, and every result must pass `audit()` before
it can be published. Failing text is retried on a stronger model, then dropped.
"""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

from . import config

_LATIN_RUN = re.compile(r"[A-Za-z][A-Za-z0-9''\-\.]{1,}")
_PERSIAN = re.compile(r"[\u0600-\u06FF]")


class LLMError(RuntimeError):
    pass


def call(prompt: str, model: str | None = None, *, system: str | None = None,
         temperature: float = 0.2, max_tokens: int = 1400, retries: int = 3) -> str:
    """Call the router. It replies with SSE even when stream=false, so the
    stream is always reassembled rather than json.load()ed (a plain json.load
    raises 'Expecting value: line 1 column 1')."""
    if not config.LLM_BASE or not config.LLM_KEY:
        raise LLMError("OPENAI_BASE_URL / OPENAI_API_KEY are not set")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    body = json.dumps({
        "model": model or config.MODEL_FAST,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }).encode("utf-8")

    last = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                config.LLM_BASE + "/chat/completions", data=body,
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {config.LLM_KEY}"})
            raw = urllib.request.urlopen(req, timeout=240).read().decode("utf-8", "replace")
            text = _parse_response(raw)
            if text.strip():
                return text.strip()
            last = "empty completion"
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode('utf-8', 'replace')}"
        except Exception as e:  # noqa: BLE001 - network layer, report and retry
            last = f"{type(e).__name__}: {e}"
        time.sleep(2 * (attempt + 1))
    raise LLMError(last or "unknown failure")


def _parse_response(raw: str) -> str:
    stripped = raw.lstrip()
    if not stripped.startswith("data:"):
        data = json.loads(raw)
        return data["choices"][0]["message"]["content"] or ""
    chunks: list[str] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload in ("", "[DONE]"):
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        for choice in data.get("choices", []):
            piece = (choice.get("delta", {}).get("content")
                     or choice.get("message", {}).get("content"))
            if piece:
                chunks.append(piece)
    return "".join(chunks)


def json_call(prompt: str, model: str | None = None, **kw) -> dict | list:
    """Call the model and parse JSON out of the reply, tolerating code fences."""
    text = call(prompt, model, **kw)
    text = re.sub(r"^\s*```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Salvage the outermost object/array — models sometimes prepend a line.
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = text.find(opener), text.rfind(closer)
            if i >= 0 and j > i:
                try:
                    return json.loads(text[i:j + 1])
                except json.JSONDecodeError:
                    continue
        raise LLMError(f"model did not return JSON: {text[:200]}")


# ── quality gate ──────────────────────────────────────────────────────────

# Letters Persian orthography never doubles at the START of a word. A word that
# begins with two of these is a garbled token, not vocabulary: live post payload
# shipped «ططمیع یک میلیارد دلاری» as a headline, which passed every existing gate
# (it is fully Persian script, so the ratio is 1.00, and it contains no Latin).
# Checked against a corpus of 2,746 unique Persian words taken from verified live
# channel posts and dry-run payloads: exactly ONE word matches, and it is that
# defect. Words like ممکن / ببرند / ممیزی start with a doubled م or ب and must
# stay legal, so the set deliberately excludes م and ب.
_NEVER_INITIAL_DOUBLE = set("طظضصغعخحذثژچگپكقفسشزرد")
_ZWNJ = "\u200c"


def _garbled(text: str) -> list[str]:
    """Persian-script words that no Persian word can be."""
    out: list[str] = []
    for word in re.findall(r"[\u0600-\u06FF\u200c]{2,}", text):
        bare = word.replace(_ZWNJ, "")
        if len(bare) >= 3 and bare[0] == bare[1] and bare[0] in _NEVER_INITIAL_DOUBLE:
            out.append(word)
    return out


# A Latin letter glued directly to a Persian LETTER inside one word. This is the
# half-transliterated proper name: live post 148 shipped «Clement-جونز» for
# Clement-Jones — half kept, half spelled out in Persian. It defeats every other
# gate for the same reason «خودregressive» did: one short fragment moves neither
# the Persian ratio nor the Latin-residue count.
#
# Persian punctuation (، ؛ ؟) sits in the Arabic Unicode block, so matching on
# the whole block would flag «OpenAI،» — a Latin word followed by a Persian
# comma, which is correct and appears in 11 of 15 live posts. The class is
# therefore restricted to LETTER ranges, and the Persian plural/possessive
# suffixes that legitimately attach to Latin acronyms (APIها، LLMهای) are
# whitelisted: measured across every live post on disk, those two rules leave
# exactly one hit, and it is the defect.
_FA_LETTER = r"\u0621-\u063A\u0641-\u064A\u0670-\u06D3"
_HALF_TRANSLIT = re.compile(
    rf"[A-Za-z][\u200c-]?[{_FA_LETTER}]|[{_FA_LETTER}][\u200c-]?[A-Za-z]")
_LATIN_WITH_FA_SUFFIX = re.compile(
    rf"^[A-Za-z][A-Za-z0-9.\-]*\u200c?(?:های|ها|ی|اش|شان)$")


def _half_transliterated(text: str) -> list[str]:
    """Words that mix Latin and Persian letters — a name translated halfway."""
    out: list[str] = []
    for token in text.split():
        core = token.strip("،؛؟.,:()[]«»\"'!؟")
        if not core or _LATIN_WITH_FA_SUFFIX.match(core):
            continue
        if _HALF_TRANSLIT.search(core):
            out.append(core)
    return out


def audit(text: str, *, min_persian_ratio: float = 0.55,
          max_latin_words: int = 6) -> tuple[bool, str]:
    """Return (ok, reason). Rejects text that is not really Persian.

    Allowed Latin: anything in config.KEEP_LATIN, version/model numbers
    (GPT-5.2, Llama-3), and acronyms up to 5 chars. Everything else counts as
    untranslated residue — the exact defect the user reported.

    Residue is checked before the ratio so the failure reason names the actual
    offending words; a bare "persian ratio 0.23" tells you nothing about which
    phrase leaked through.

    Garbled Persian is checked first of all, because it is invisible to both of
    the other tests: a mangled word is still Persian script, so the ratio reads
    1.00 and there is no Latin residue to find.
    """
    if not text or not text.strip():
        return False, "empty"

    garbled = _garbled(text)
    if garbled:
        return False, f"garbled Persian: {garbled[:5]}"

    half = _half_transliterated(text)
    if half:
        return False, f"half-transliterated: {half[:5]}"

    allowed = {w.lower() for w in config.KEEP_LATIN}
    residue: list[str] = []
    for word in _LATIN_RUN.findall(text):
        bare = word.strip(".,;:!?()[]{}\"'").rstrip("-")
        if not bare:
            continue
        low = bare.lower()
        if low in allowed:
            continue
        # brand + version: GPT-5.2, Llama-3, o3-mini, Claude-4
        head = re.split(r"[-_.\d]", bare)[0]
        if head.lower() in allowed:
            continue
        if len(bare) <= 5 and bare.isupper():
            continue          # acronym such as SDK, IDE
        if re.fullmatch(r"[A-Za-z]\d+", bare):
            continue          # o3, v2
        residue.append(bare)

    if len(residue) > max_latin_words:
        return False, f"untranslated: {residue[:8]}"

    persian = len(_PERSIAN.findall(text))
    letters = len(re.findall(r"[^\W\d_]", text, re.UNICODE))
    if letters == 0:
        return False, "no letters"
    ratio = persian / letters
    if ratio < min_persian_ratio:
        return False, f"persian ratio {ratio:.2f} < {min_persian_ratio}"

    return True, "ok"
