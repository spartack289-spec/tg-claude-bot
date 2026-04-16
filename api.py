import os
import hmac
import hashlib
import json
import time
from contextlib import asynccontextmanager
from urllib.parse import parse_qs, unquote

import httpx
import anthropic
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from agents import AGENTS, AGENTS_BY_ID
from collector import collect_and_save, load_digests

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PROXY_URL = os.getenv("PROXY_URL")
ALLOWED_USER_IDS = set(
    int(uid.strip()) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()
)
MODEL = "claude-sonnet-4-6"
MAX_HISTORY = 20

# In-memory history: { (user_id, agent_id): [{role, content}, ...] }
mini_app_histories: dict[tuple, list] = {}

claude: anthropic.AsyncAnthropic | None = None
claude_direct: anthropic.AsyncAnthropic | None = None
_proxy_http_client: httpx.AsyncClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global claude, claude_direct, _proxy_http_client

    # Initialize Anthropic client(s) inside the running event loop
    if PROXY_URL:
        _proxy_http_client = httpx.AsyncClient(proxy=PROXY_URL)
        claude = anthropic.AsyncAnthropic(
            api_key=ANTHROPIC_API_KEY,
            http_client=_proxy_http_client,
        )
    else:
        claude = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Always keep a direct (no-proxy) client as fallback
    claude_direct = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Start scheduler
    scheduler = AsyncIOScheduler(timezone="Europe/Moscow")
    scheduler.add_job(collect_and_save, "cron", hour="8,14,20", minute=0)
    scheduler.start()
    print("[api] Планировщик запущен: дайджест в 8:00, 14:00, 20:00 МСК")

    yield

    # Cleanup
    scheduler.shutdown(wait=False)
    if _proxy_http_client:
        await _proxy_http_client.aclose()
    await claude.close()
    if PROXY_URL:
        await claude_direct.close()


async def _call_claude(**kwargs):
    """Call Claude API, falling back to direct connection if proxy fails."""
    try:
        return await claude.messages.create(**kwargs)
    except anthropic.APIConnectionError:
        if PROXY_URL:
            print("[api] Прокси недоступен, пробую прямое соединение...")
            return await claude_direct.messages.create(**kwargs)
        raise


app = FastAPI(lifespan=lifespan)


def verify_init_data(init_data: str) -> dict:
    """Validates Telegram WebApp initData. Returns user dict on success."""
    if not init_data:
        raise HTTPException(status_code=401, detail="Missing initData")

    parsed = parse_qs(init_data, keep_blank_values=True)

    received_hash = parsed.pop("hash", [None])[0]
    if not received_hash:
        raise HTTPException(status_code=401, detail="Missing hash in initData")

    data_check_string = "\n".join(
        sorted(f"{k}={v[0]}" for k, v in parsed.items())
    )

    secret_key = hmac.new(b"WebAppData", BOT_TOKEN.encode(), hashlib.sha256).digest()
    expected_hash = hmac.new(secret_key, data_check_string.encode(), hashlib.sha256).hexdigest()

    if not hmac.compare_digest(received_hash, expected_hash):
        raise HTTPException(status_code=401, detail="Invalid initData signature")

    auth_date = int(parsed.get("auth_date", [0])[0])
    if time.time() - auth_date > 3600:
        raise HTTPException(status_code=401, detail="initData expired")

    user_json = parsed.get("user", [None])[0]
    if not user_json:
        raise HTTPException(status_code=401, detail="No user in initData")

    return json.loads(unquote(user_json))


class ChatRequest(BaseModel):
    agent_id: str
    message: str
    telegram_init_data: str


@app.get("/")
async def serve_index():
    return FileResponse("static/index.html")


@app.get("/api/agents")
async def get_agents():
    return [
        {
            "id": a["id"],
            "name": a["name"],
            "emoji": a["emoji"],
            "description": a["description"],
            "type": a.get("type", "chat"),
        }
        for a in AGENTS
    ]


@app.get("/api/news")
async def get_news():
    return load_digests()


@app.post("/api/news/refresh")
async def refresh_news(request: Request):
    init_data = request.headers.get("X-Telegram-Init-Data", "")
    user = verify_init_data(init_data)
    if ALLOWED_USER_IDS and user["id"] not in ALLOWED_USER_IDS:
        raise HTTPException(status_code=403, detail="Access denied")
    try:
        await collect_and_save()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {"status": "ok", "digests": load_digests()}


@app.post("/api/chat")
async def chat(req: ChatRequest):
    user = verify_init_data(req.telegram_init_data)
    user_id = user["id"]

    if ALLOWED_USER_IDS and user_id not in ALLOWED_USER_IDS:
        raise HTTPException(status_code=403, detail="Access denied")

    agent = AGENTS_BY_ID.get(req.agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    key = (user_id, req.agent_id)
    if key not in mini_app_histories:
        mini_app_histories[key] = []

    history = mini_app_histories[key]
    history.append({"role": "user", "content": req.message})

    if len(history) > MAX_HISTORY:
        history[:] = history[-MAX_HISTORY:]

    try:
        response = await _call_claude(
            model=MODEL,
            max_tokens=1024,
            system=agent["system_prompt"],
            messages=history,
        )
        reply = response.content[0].text
    except anthropic.APIConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Нет соединения с AI: {e}")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ошибка AI API: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    history.append({"role": "assistant", "content": reply})

    return {"reply": reply, "agent_id": req.agent_id}


@app.delete("/api/chat/{agent_id}/history")
async def clear_history(agent_id: str, request: Request):
    init_data = request.headers.get("X-Telegram-Init-Data", "")
    user = verify_init_data(init_data)
    key = (user["id"], agent_id)
    mini_app_histories.pop(key, None)
    return {"status": "cleared"}
