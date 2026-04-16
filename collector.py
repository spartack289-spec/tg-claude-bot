import os
import json
import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import httpx
import feedparser
import anthropic
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
PROXY_URL = os.getenv("PROXY_URL")
DIGESTS_FILE = "digests.json"
MAX_DIGESTS = 10
MODEL = "claude-sonnet-4-6"

MOSCOW_TZ = ZoneInfo("Europe/Moscow")

RSS_FEEDS = [
    ("TechCrunch AI", "https://techcrunch.com/category/artificial-intelligence/feed/"),
    ("VentureBeat AI", "https://venturebeat.com/category/ai/feed/"),
    ("The Verge AI", "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml"),
    ("Wired AI", "https://www.wired.com/feed/tag/ai/latest/rss"),
]

ARXIV_RSS = "https://export.arxiv.org/rss/cs.AI"
PRODUCTHUNT_RSS = "https://www.producthunt.com/feed"


def load_digests() -> list:
    if os.path.exists(DIGESTS_FILE):
        with open(DIGESTS_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_digests(digests: list):
    with open(DIGESTS_FILE, "w", encoding="utf-8") as f:
        json.dump(digests, f, ensure_ascii=False, indent=2)


def fetch_rss(url: str, max_items: int = 5) -> list[dict]:
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", entry.get("description", ""))[:300],
                "link": entry.get("link", ""),
            })
        return items
    except Exception:
        return []


def fetch_arxiv(max_items: int = 5) -> list[dict]:
    try:
        feed = feedparser.parse(ARXIV_RSS)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", ""),
                "summary": entry.get("summary", "")[:300],
                "link": entry.get("link", ""),
                "source": "arxiv",
            })
        return items
    except Exception:
        return []


def fetch_producthunt(max_items: int = 5) -> list[dict]:
    try:
        feed = feedparser.parse(PRODUCTHUNT_RSS)
        items = []
        for entry in feed.entries[:max_items]:
            title = entry.get("title", "").lower()
            summary = entry.get("summary", "").lower()
            if any(kw in title + summary for kw in ["ai", "ml", "gpt", "llm", "model", "agent", "automation"]):
                items.append({
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", "")[:300],
                    "link": entry.get("link", ""),
                    "source": "producthunt",
                })
        return items[:max_items]
    except Exception:
        return []


async def fetch_web_search(query: str, max_results: int = 5) -> list[dict]:
    if not TAVILY_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            res = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": TAVILY_API_KEY,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max_results,
                    "include_answer": False,
                },
            )
            data = res.json()
            return [
                {
                    "title": r.get("title", ""),
                    "summary": r.get("content", "")[:300],
                    "link": r.get("url", ""),
                    "source": "web",
                }
                for r in data.get("results", [])
            ]
    except Exception:
        return []


async def generate_digest(items: list[dict]) -> tuple[str, list[dict]]:
    """Generate digest text and links list using Claude."""
    news_text = ""
    links = []
    for i, item in enumerate(items, 1):
        news_text += f"{i}. {item['title']}\n{item['summary']}\nСсылка: {item['link']}\n\n"
        if item.get("link"):
            links.append({"title": item["title"], "url": item["link"]})

    client = anthropic.AsyncAnthropic(
        api_key=ANTHROPIC_API_KEY,
        http_client=httpx.AsyncClient(proxy=PROXY_URL) if PROXY_URL else None,
    )

    prompt = f"""Тебе дан список новостей об искусственном интеллекте и технологических стартапах.
Напиши краткий дайджест на русском языке в 3 разделах:

🔬 **Достижения** — новые модели, прорывы, анонсы крупных компаний (3-5 пунктов)
💡 **Стартапы** — инвестиции, запуски новых продуктов, интересные компании (3-5 пунктов)
📄 **Наука** — arxiv-статьи, исследования (2-3 пункта)

Каждый пункт — 1-2 предложения по сути. Пиши живо и по делу, без воды.
Если раздел пустой — пропусти его.

Новости:
{news_text}"""

    response = await client.messages.create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text
    return content, links[:15]


def get_digest_title() -> str:
    now = datetime.now(MOSCOW_TZ)
    hour = now.hour
    date_str = now.strftime("%-d %B").replace(
        "January", "января").replace("February", "февраля").replace(
        "March", "марта").replace("April", "апреля").replace(
        "May", "мая").replace("June", "июня").replace(
        "July", "июля").replace("August", "августа").replace(
        "September", "сентября").replace("October", "октября").replace(
        "November", "ноября").replace("December", "декабря")
    if hour < 12:
        return f"Утренний дайджест · {date_str}"
    elif hour < 18:
        return f"Дневной дайджест · {date_str}"
    else:
        return f"Вечерний дайджест · {date_str}"


async def collect_and_save():
    """Main function: collect news, generate digest, save."""
    print(f"[collector] Сбор новостей... {datetime.now(MOSCOW_TZ).strftime('%H:%M')}")

    items = []

    # RSS feeds
    for name, url in RSS_FEEDS:
        rss_items = fetch_rss(url, max_items=4)
        for item in rss_items:
            item["source"] = name
        items.extend(rss_items)

    # ArXiv
    items.extend(fetch_arxiv(max_items=4))

    # ProductHunt
    items.extend(fetch_producthunt(max_items=4))

    # Web search
    web_items = await fetch_web_search("latest AI news breakthroughs today", max_results=5)
    items.extend(web_items)
    startup_items = await fetch_web_search("AI startups funding raised 2026", max_results=3)
    items.extend(startup_items)

    if not items:
        print("[collector] Нет данных для дайджеста")
        return

    # Remove duplicates by title
    seen = set()
    unique_items = []
    for item in items:
        key = item["title"][:60]
        if key not in seen:
            seen.add(key)
            unique_items.append(item)

    print(f"[collector] Собрано {len(unique_items)} уникальных новостей")

    content, links = await generate_digest(unique_items[:30])

    now = datetime.now(MOSCOW_TZ)
    digest = {
        "id": now.strftime("%Y-%m-%dT%H:%M"),
        "created_at": now.isoformat(),
        "title": get_digest_title(),
        "content": content,
        "links": links,
    }

    digests = load_digests()
    digests.insert(0, digest)
    digests = digests[:MAX_DIGESTS]
    save_digests(digests)

    print(f"[collector] Дайджест сохранён: {digest['title']}")


if __name__ == "__main__":
    asyncio.run(collect_and_save())
