"""Configuration for AI Radar (رادار هوش مصنوعی).

Every feed here was probed live before being added — see tests/probe_feeds.py.
Dead feeds are kept in DEAD_FEEDS with the observed HTTP status so nobody
re-adds them by guesswork.
"""
from __future__ import annotations

import os

# ── Telegram ──────────────────────────────────────────────────────────────
BOT_TOKEN = os.environ.get("TG_BOT_TOKEN", "")
CHANNEL_ID = os.environ.get("TG_CHANNEL_ID", "")

# ── LLM (OpenAI-compatible router) ────────────────────────────────────────
LLM_BASE = os.environ.get("OPENAI_BASE_URL", "").rstrip("/")
LLM_KEY = os.environ.get("OPENAI_API_KEY", "")

# geminifl answered a full news paragraph in 1.2s with 6 Latin tokens; nv took
# 15.8s and transliterated brand names into Persian letters ("اوپن‌ای‌آی"),
# which reads worse for a tech audience. geminifl is the workhorse; gemini is
# the escalation path when the quality gate rejects a translation.
MODEL_FAST = os.environ.get("RADAR_MODEL_FAST", "geminifl")
MODEL_STRONG = os.environ.get("RADAR_MODEL_STRONG", "gemini")

# ── Sources ───────────────────────────────────────────────────────────────
# tier: 1 = primary lab/vendor announcement, 2 = quality press, 3 = research firehose
FEEDS: list[dict] = [
    {"name": "OpenAI",        "url": "https://openai.com/blog/rss.xml",                              "tier": 1, "fa": "اوپن‌ای‌آی"},
    {"name": "DeepMind",      "url": "https://deepmind.google/blog/rss.xml",                          "tier": 1, "fa": "دیپ‌مایند"},
    {"name": "Google AI",     "url": "https://blog.google/technology/ai/rss/",                        "tier": 1, "fa": "گوگل"},
    {"name": "Hugging Face",  "url": "https://huggingface.co/blog/feed.xml",                          "tier": 1, "fa": "هاگینگ‌فیس"},
    {"name": "Mistral",       "url": "https://mistral.ai/rss.xml",                                    "tier": 1, "fa": "میسترال"},
    {"name": "NVIDIA",        "url": "https://blogs.nvidia.com/feed/",                                "tier": 1, "fa": "انویدیا"},
    {"name": "Microsoft",     "url": "https://news.microsoft.com/source/feed/",                       "tier": 1, "fa": "مایکروسافت"},
    {"name": "Meta",          "url": "https://about.fb.com/news/feed/",                               "tier": 1, "fa": "متا"},

    {"name": "MIT Tech Review", "url": "https://www.technologyreview.com/feed/",                      "tier": 2, "fa": "ام‌آی‌تی تکنالوجی ریویو"},
    {"name": "TechCrunch",    "url": "https://techcrunch.com/tag/artificial-intelligence/feed/",      "tier": 2, "fa": "تک‌کرانچ"},
    {"name": "The Verge",     "url": "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "tier": 2, "fa": "دی‌ورج"},
    {"name": "VentureBeat",   "url": "https://venturebeat.com/category/ai/feed/",                     "tier": 2, "fa": "ونچربیت"},
    {"name": "Ars Technica",  "url": "https://feeds.arstechnica.com/arstechnica/technology-lab",      "tier": 2, "fa": "آرس تکنیکا"},
    {"name": "Wired",         "url": "https://www.wired.com/feed/tag/ai/latest/rss",                  "tier": 2, "fa": "وایرد"},
    {"name": "Stratechery",   "url": "https://stratechery.com/feed/",                                 "tier": 2, "fa": "استرتیکری"},
    {"name": "Simon Willison","url": "https://simonwillison.net/atom/everything/",                    "tier": 2, "fa": "سایمون ویلیسون"},
    {"name": "Import AI",     "url": "https://importai.substack.com/feed",                            "tier": 2, "fa": "ایمپورت ای‌آی"},
    {"name": "Sequoia",       "url": "https://www.sequoiacap.com/feed/",                              "tier": 2, "fa": "سکویا"},
    {"name": "Hacker News",   "url": "https://hnrss.org/frontpage?points=150",                        "tier": 2, "fa": "هکر نیوز"},

    {"name": "arXiv cs.AI",   "url": "http://export.arxiv.org/rss/cs.AI",                             "tier": 3, "fa": "آرکایو"},
    {"name": "arXiv cs.LG",   "url": "http://export.arxiv.org/rss/cs.LG",                             "tier": 3, "fa": "آرکایو"},
    {"name": "arXiv cs.CL",   "url": "http://export.arxiv.org/rss/cs.CL",                             "tier": 3, "fa": "آرکایو"},
]

# Probed and confirmed broken — do not resurrect without re-probing.
DEAD_FEEDS = {
    "https://www.anthropic.com/news/rss.xml": "404",
    "https://www.anthropic.com/rss.xml": "404",
    "https://ai.meta.com/blog/rss/": "400",
    "https://blogs.microsoft.com/ai/feed/": "410 Gone",
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml": "404 (path moved)",
    "https://bair.berkeley.edu/blog/feed.xml": "unreachable",
    "https://www.zdnet.com/topic/artificial-intelligence/rss.xml": "unreachable",
    "https://stability.ai/news?format=rss": "unreachable",
    "https://ai.googleblog.com/feeds/posts/default": "retired",
}

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# ── Editorial shape ───────────────────────────────────────────────────────
MAX_STORIES = 9           # per bulletin; more than this and the toggles get unreadable
MAX_PER_SECTION = 3
# How far the backfill may exceed MAX_PER_SECTION when other sections are empty.
# Post 110 shipped 6 of 9 stories from `models` (eight arXiv cards) because the
# backfill ignored the cap outright; a widened window is mostly arXiv.
BACKFILL_SLACK = 1
LOOKBACK_HOURS = 8        # slight overlap with the 6h cadence so nothing slips through
# A bulletin with one story is worse than no bulletin: measured live on channel
# post 106, where 8 of 9 stories in the window had already been published and
# the post shipped a single item. When the fresh count falls under this floor
# the window is widened (see run.py) instead of publishing something thin.
MIN_STORIES = 4
WIDEN_LADDER = (24, 72, 168)   # hours, tried in order when the window is thin
STATE_PATH = os.environ.get("RADAR_STATE", "data/seen.json")
STATE_KEEP = 4000         # remembered fingerprints

SECTIONS = [
    ("models",   "🔬 مدل‌ها و پژوهش"),
    ("business", "💰 کسب‌وکار و سرمایه"),
    ("policy",   "🏛 سیاست و قانون"),
    ("tools",    "🛠 ابزار و کد"),
]

# Brand names that must stay in Latin script. The old project transliterated
# these ("جی‌پی‌یو" for GPT — a real bug in its mapping table) which is exactly
# the kind of error the quality gate now rejects.
KEEP_LATIN = {
    "OpenAI", "GPT", "ChatGPT", "Anthropic", "Claude", "Google", "DeepMind", "Gemini",
    "Meta", "Llama", "Mistral", "NVIDIA", "AMD", "Intel", "Microsoft", "Copilot",
    "Apple", "Amazon", "AWS", "Azure", "Hugging Face", "HuggingFace", "Qwen",
    "DeepSeek", "Grok", "xAI", "Midjourney", "Stability", "Cohere", "Perplexity",
    "Sora", "Whisper", "PyTorch", "TensorFlow", "CUDA", "Transformer", "MoE",
    "API", "LLM", "RAG", "AGI", "GPU", "TPU", "MMLU", "GPQA", "HumanEval", "SWE-bench",
    "arXiv", "GitHub", "Linux", "iOS", "Android", "Windows", "Nature", "Science",
}
