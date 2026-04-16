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
FETCH_TIMEOUT = 12  # seconds per RSS request

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


async def fetch_rss_async(name: str, url: str, max_items: int = 5) -> list[dict]:
    """Fetch RSS feed using httpx with timeout, then parse with feedparser."""
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; RSS reader)"})
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        items = []
        for entry in feed.entries[:max_items]:
            items.append({
                "title": entry.get("title", "").strip(),
                "summary": entry.get("summary", entry.get("description", ""))[:400].strip(),
                "link": entry.get("link", ""),
                "source": name,
            })
        print(f"[collector] {name}: {len(items)} статей")
        return items
    except Exception as e:
        print(f"[collector] {name}: ошибка — {e}")
        return []


async def fetch_arxiv() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(ARXIV_RSS)
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        items = []
        for entry in feed.entries[:6]:
            items.append({
                "title": entry.get("title", "").strip(),
                "summary": entry.get("summary", "")[:400].strip(),
                "link": entry.get("link", ""),
                "source": "arxiv",
            })
        print(f"[collector] arxiv: {len(items)} статей")
        return items
    except Exception as e:
        print(f"[collector] arxiv: ошибка — {e}")
        return []


async def fetch_producthunt() -> list[dict]:
    try:
        async with httpx.AsyncClient(timeout=FETCH_TIMEOUT, follow_redirects=True) as client:
            resp = await client.get(PRODUCTHUNT_RSS, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
        feed = feedparser.parse(resp.text)
        items = []
        for entry in feed.entries[:20]:
            title = entry.get("title", "").lower()
            summary = entry.get("summary", "").lower()
            if any(kw in title + summary for kw in ["ai", "ml", "gpt", "llm", "model", "agent", "automation", "neural"]):
                items.append({
                    "title": entry.get("title", "").strip(),
                    "summary": entry.get("summary", "")[:400].strip(),
                    "link": entry.get("link", ""),
                    "source": "producthunt",
                })
        result = items[:5]
        print(f"[collector] producthunt: {len(result)} продуктов")
        return result
    except Exception as e:
        print(f"[collector] producthunt: ошибка — {e}")
        return []


async def fetch_web_search(query: str, max_results: int = 5) -> list[dict]:
    if not TAVILY_API_KEY:
        print("[collector] Tavily: TAVILY_API_KEY не задан, пропускаю")
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
            res.raise_for_status()
            data = res.json()
            results = [
                {
                    "title": r.get("title", "").strip(),
                    "summary": r.get("content", "")[:400].strip(),
                    "link": r.get("url", ""),
                    "source": "web",
                }
                for r in data.get("results", [])
            ]
            print(f"[collector] Tavily «{query[:40]}»: {len(results)} результатов")
            return results
    except Exception as e:
        print(f"[collector] Tavily: ошибка — {e}")
        return []


async def _anthropic_create(**kwargs) -> anthropic.types.Message:
    """Call Claude API with proxy, falling back to direct connection if proxy fails."""
    if PROXY_URL:
        proxy_http = httpx.AsyncClient(proxy=PROXY_URL)
        try:
            client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY, http_client=proxy_http)
            try:
                return await client.messages.create(**kwargs)
            except anthropic.APIConnectionError:
                print("[collector] Прокси недоступен, пробую прямое соединение...")
            finally:
                await client.close()
        finally:
            await proxy_http.aclose()

    # Direct connection (no proxy or proxy failed)
    async with httpx.AsyncClient() as http:
        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY, http_client=http)
        try:
            return await client.messages.create(**kwargs)
        finally:
            await client.close()


async def generate_digest(items: list[dict]) -> tuple[str, list[dict]]:
    """Generate digest text and links list using Claude."""
    news_text = ""
    links = []
    for i, item in enumerate(items, 1):
        news_text += f"{i}. [{item.get('source', '')}] {item['title']}\n{item['summary']}\nСсылка: {item['link']}\n\n"
        if item.get("link"):
            links.append({"title": item["title"], "url": item["link"]})

    prompt = f"""Тебе дан список новостей об искусственном интеллекте и технологических стартапах.
Напиши краткий дайджест на русском языке в 3 разделах:

🔬 **Достижения** — новые модели, прорывы, анонсы крупных компаний (3-5 пунктов)
💡 **Стартапы** — инвестиции, запуски новых продуктов, интересные компании (3-5 пунктов)
📄 **Наука** — arxiv-статьи, исследования (2-3 пункта)

Каждый пункт — 1-2 предложения по сути. Пиши живо и по делу, без воды.
Если раздел пустой — пропусти его.

Новости:
{news_text}"""

    response = await _anthropic_create(
        model=MODEL,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    content = response.content[0].text
    return content, links[:15]


def get_digest_title() -> str:
    now = datetime.now(MOSCOW_TZ)
    hour = now.hour
    months = ["января","февраля","марта","апреля","мая","июня",
              "июля","августа","сентября","октября","ноября","декабря"]
    date_str = f"{now.day} {months[now.month - 1]}"
    if hour < 12:
        return f"Утренний дайджест · {date_str}"
    elif hour < 18:
        return f"Дневной дайджест · {date_str}"
    else:
        return f"Вечерний дайджест · {date_str}"


async def collect_and_save():
    """Main function: collect news from all sources in parallel, generate digest, save."""
    now_str = datetime.now(MOSCOW_TZ).strftime("%H:%M %d.%m")
    print(f"[collector] Начинаю сбор новостей... {now_str}")


    # Run all RSS fetches in parallel
    rss_tasks = [fetch_rss_async(name, url, max_items=5) for name, url in RSS_FEEDS]
    rss_results = await asyncio.gather(*rss_tasks)

    items: list[dict] = []
    for batch in rss_results:
        items.extend(batch)

    # ArXiv + ProductHunt in parallel
    arxiv_items, ph_items = await asyncio.gather(fetch_arxiv(), fetch_producthunt())
    items.extend(arxiv_items)
    items.extend(ph_items)

    # Web search via Tavily
    web_tasks = [
        fetch_web_search("AI artificial intelligence news this week", max_results=5),
        fetch_web_search("AI startups funding investment 2025 2026", max_results=4),
        fetch_web_search("new AI model release announcement", max_results=3),
    ]
    web_results = await asyncio.gather(*web_tasks)
    for batch in web_results:
        items.extend(batch)

    if not items:
        print("[collector] Нет данных ни из одного источника — дайджест не создан")
        return

    # Remove duplicates by title prefix
    seen: set[str] = set()
    unique_items: list[dict] = []
    for item in items:
        key = item["title"][:60].lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique_items.append(item)

    print(f"[collector] Итого уникальных статей: {len(unique_items)}")

    try:
        content, links = await generate_digest(unique_items[:35])
    except Exception as e:
        print(f"[collector] Ошибка генерации дайджеста: {e}")
        raise RuntimeError(f"Не удалось сгенерировать дайджест: {e}") from e

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

    print(f"[collector] Готово: «{digest['title']}» ({len(links)} ссылок)")


if __name__ == "__main__":
    asyncio.run(collect_and_save())
