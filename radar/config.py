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

    # Added because the pool was 120/143 arXiv, and arXiv carries NO article art:
    # a balanced-by-section bulletin still came out as nine pictureless abstracts
    # (post 112). Every feed below was probed live for both yield and images —
    # 16/16 sampled articles shipped an inline image that passed the Telegram
    # usability check, all 16 distinct. Yield is AI-related items per 72h.
    {"name": "AWS ML",         "url": "https://aws.amazon.com/blogs/machine-learning/feed/",          "tier": 1, "fa": "ای‌دبلیواس"},           # 17/72h
    {"name": "The Decoder",    "url": "https://the-decoder.com/feed/",                                "tier": 2, "fa": "دیکودر"},              # 10/72h
    {"name": "Semafor Tech",   "url": "https://www.semafor.com/rss.xml",                              "tier": 2, "fa": "سمافور"},              # 10/72h
    {"name": "AI Business",    "url": "https://aibusiness.com/rss.xml",                               "tier": 2, "fa": "ای‌آی بیزنس"},          # 6/72h
    {"name": "IEEE Spectrum",  "url": "https://spectrum.ieee.org/feeds/topic/artificial-intelligence.rss", "tier": 2, "fa": "آی‌تریپل‌ای اسپکتروم"},  # 2/72h
    {"name": "MIT News",       "url": "https://news.mit.edu/rss/topic/artificial-intelligence2",      "tier": 2, "fa": "ام‌آی‌تی نیوز"},        # 2/72h

    # Third sweep (40 more candidates), added because the pool went one-sided
    # again: post 131 shipped 5 stories and ONE photo from a 93-item unseen pool
    # that was 92 arXiv. The problem is not the renderer or the caps — it is that
    # the non-arXiv feeds are all high-signal and low-volume, so a day with 28
    # already-published items leaves nothing but abstracts. These four were
    # validated through radar.sources itself (fetch_feed + is_ai_related), not by
    # HTTP status: counts are AI items in the last 48h / how many carry art.
    {"name": "SiliconANGLE",   "url": "https://siliconangle.com/category/ai/feed/",                 "tier": 2, "fa": "سیلیکون‌انگل"},      # 16 AI/48h, 16 with art
    {"name": "Tom's Hardware", "url": "https://www.tomshardware.com/feeds/all",                     "tier": 2, "fa": "تامز هاردور"},       # 19 AI/48h, 19 with art
    {"name": "TechRadar AI",   "url": "https://www.techradar.com/feeds/tag/ai",                     "tier": 2, "fa": "تک‌ریدار"},          # 15 AI/48h, 15 with art
    {"name": "Ars Technica AI","url": "https://arstechnica.com/ai/feed/",                           "tier": 2, "fa": "آرس تکنیکا"},        # 4 AI/48h, 4 with art

    {"name": "arXiv cs.AI",   "url": "http://export.arxiv.org/rss/cs.AI",                             "tier": 3, "fa": "آرکایو"},
    {"name": "arXiv cs.LG",   "url": "http://export.arxiv.org/rss/cs.LG",                             "tier": 3, "fa": "آرکایو"},
    {"name": "arXiv cs.CL",   "url": "http://export.arxiv.org/rss/cs.CL",                             "tier": 3, "fa": "آرکایو"},
]

# Probed and confirmed broken — do not resurrect without re-probing.
DEAD_FEEDS = {
    "https://www.anthropic.com/news/rss.xml": "404",
    "https://www.anthropic.com/rss.xml": "404",
    "https://www.anthropic.com/engineering/rss.xml": "404",
    "https://www.anthropic.com/news.xml": "404",
    "https://ai.meta.com/blog/rss/": "400",
    "https://blogs.microsoft.com/ai/feed/": "410 Gone",
    "https://www.theverge.com/ai-artificial-intelligence/rss/index.xml": "404 (path moved)",
    "https://bair.berkeley.edu/blog/feed.xml": "unreachable",
    "https://www.zdnet.com/topic/artificial-intelligence/rss.xml": "unreachable",
    "https://stability.ai/news?format=rss": "unreachable",
    "https://ai.googleblog.com/feeds/posts/default": "retired",
    # Second sweep (40 candidates probed): HTTP 200 but zero parseable items,
    # i.e. an HTML page or a JS shell, not a feed. Do not re-add on the strength
    # of a 200.
    "https://analyticsindiamag.com/feed/": "200 but 0 items",
    "https://cohere.com/blog/rss.xml": "200 but 0 items",
    "https://blog.langchain.dev/rss/": "200 but 0 items",
    "https://hai.stanford.edu/news/rss.xml": "200 but 0 items",
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss25&id=19854910": "200 but 0 items",
    "https://www.marktechpost.com/feed/": "202 (bot wall)",
    "https://www.engadget.com/rss.xml": "403",
    "https://www.perplexity.ai/hub/blog/rss.xml": "403",
    "https://groq.com/feed/": "404",
    "https://blog.vllm.ai/feed.xml": "404",
    "https://runwayml.com/blog/rss.xml": "404",
    "https://elevenlabs.io/blog/rss.xml": "404",
    "https://www.qualcomm.com/news/releases.rss": "404",
    "https://api.axios.com/feed/technology": "404",
    "https://stability.ai/blog?format=rss": "404",
    # Live but useless in practice: parseable feed, zero AI items in 72h.
    "https://www.together.ai/blog/rss.xml": "0 AI items/72h",
    "https://replicate.com/blog/rss": "0 AI items/72h",
    "https://ollama.com/blog/rss.xml": "0 AI items/72h",
    "https://www.nature.com/natmachintell.rss": "0 AI items/72h",
    "https://syncedreview.com/feed/": "0 AI items/72h",
    "https://thegradient.pub/rss/": "0 AI items/72h",
    "https://www.interconnects.ai/feed": "0 AI items/72h",
    "https://lastweekin.ai/feed": "0 AI items/72h",
    # Third sweep. Note the pattern: vendor blogs behind a JS shell answer 200
    # with zero items, and news feeds that DO parse can still be useless here.
    "https://research.google/blog/rss/": "0 AI items/48h (2 posts, neither AI)",
    "https://www.theguardian.com/technology/artificialintelligenceai/rss": "16 AI/48h but 0 images",
    "https://feeds.bbci.co.uk/news/technology/rss.xml": "2 AI/48h — too thin",
    "https://www.theregister.com/software/ai_ml/headlines.atom": "1 AI/48h — too thin",
    "https://openai.com/news/rss.xml": "648 entries, 0 items parse (sitemap-style)",
    "https://qwenlm.github.io/blog/index.xml": "no art in any entry",
    "https://feeds.bloomberg.com/technology/news.rss": "0 art, paywalled bodies",
    "https://www.ft.com/technology?format=rss": "0 art, paywalled bodies",
    "https://www.reutersagency.com/feed/?best-topics=tech&post_type=best": "HTTPError",
    "https://www.engadget.com/tag/ai/rss.xml": "HTTPError",
    "https://www.databricks.com/blog/feed": "HTTPError",
    "https://modal.com/blog/feed.xml": "HTTPError",
    "https://mistral.ai/news/feed.xml": "HTTPError",
}

USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")

# ── Editorial shape ───────────────────────────────────────────────────────
MAX_STORIES = 9           # per bulletin; more than this and the toggles get unreadable
MAX_PER_SECTION = 3
# One source family (all three arXiv feeds count as one) may not swallow the
# bulletin either. Post 112 was perfectly balanced by SECTION and still shipped
# nine research abstracts with zero photos, because arXiv papers land in every
# section and arXiv carries no article art. Measured: the 24h pool is 143
# stories, 120 of them arXiv.
MAX_PER_FAMILY = 3
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

# Which theme the NEXT bulletin uses. A separate file from the fingerprints
# because it is a counter, not a set: `save_seen` prunes by value, so a counter
# living in seen.json would eventually be pruned away as if it were a stale
# fingerprint.
ROTATION_PATH = os.environ.get("RADAR_ROTATION", "data/rotation.json")

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
