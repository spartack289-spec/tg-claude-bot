import asyncio
import os
import json
import base64
import io
from dotenv import load_dotenv
import anthropic
import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup, InlineKeyboardButton
from aiogram.client.session.aiohttp import AiohttpSession

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
PROXY_URL = os.getenv("PROXY_URL")
ALLOWED_USER_IDS = set(
    int(uid.strip()) for uid in os.getenv("ALLOWED_USER_IDS", "").split(",") if uid.strip()
)

HISTORY_FILE = "history.json"
DEFAULT_MODEL = "claude-sonnet-4-6"
MODELS = {
    "sonnet": ("claude-sonnet-4-6", "Sonnet (умный)"),
    "haiku": ("claude-haiku-4-5-20251001", "Haiku (быстрый)"),
}

dp = Dispatcher()
claude = anthropic.Anthropic(
    api_key=ANTHROPIC_API_KEY,
    http_client=httpx.Client(proxy=PROXY_URL) if PROXY_URL else None
)

chat_histories: dict[int, list] = {}
user_models: dict[int, str] = {}


def load_history():
    global chat_histories
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
        chat_histories = {int(k): v for k, v in data.items()}


def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in chat_histories.items()}, f, ensure_ascii=False)


def is_allowed(user_id: int) -> bool:
    if not ALLOWED_USER_IDS:
        return True
    return user_id in ALLOWED_USER_IDS


@dp.message(CommandStart())
async def cmd_start(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("Доступ запрещён")
        return
    chat_histories[message.from_user.id] = []
    save_history()
    await message.answer(
        "Привет! Я бот на базе Claude. Напиши мне что-нибудь — отвечу на любой вопрос.\n\n"
        "Команды:\n"
        "/start — начать заново\n"
        "/clear — очистить историю\n"
        "/model — выбрать модель\n"
        "/myid — узнать свой Telegram ID"
    )


@dp.message(Command("clear"))
async def cmd_clear(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("Доступ запрещён")
        return
    chat_histories[message.from_user.id] = []
    save_history()
    await message.answer("История очищена. Начинаем с чистого листа!")


@dp.message(Command("myid"))
async def cmd_myid(message: Message):
    await message.answer(f"Твой Telegram ID: {message.from_user.id}")


@dp.message(Command("model"))
async def cmd_model(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("Доступ запрещён")
        return
    current = user_models.get(message.from_user.id, DEFAULT_MODEL)
    keyboard = InlineKeyboardMarkup(inline_keyboard=[[
        InlineKeyboardButton(text="Sonnet (умный)", callback_data="model:sonnet"),
        InlineKeyboardButton(text="Haiku (быстрый)", callback_data="model:haiku"),
    ]])
    await message.answer(f"Текущая модель: {current}\n\nВыбери новую:", reply_markup=keyboard)


@dp.callback_query(lambda c: c.data and c.data.startswith("model:"))
async def handle_model_callback(callback: CallbackQuery):
    key = callback.data.split(":")[1]
    if key in MODELS:
        model_id, model_name = MODELS[key]
        user_models[callback.from_user.id] = model_id
        await callback.answer(f"Переключено: {model_name}")
        await callback.message.edit_text(f"Модель выбрана: {model_name}")


@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
    if not is_allowed(message.from_user.id):
        await message.answer("Доступ запрещён")
        return

    await bot.send_chat_action(message.chat.id, "typing")

    photo = message.photo[-1]
    file = await bot.get_file(photo.file_id)
    buffer = io.BytesIO()
    await bot.download_file(file.file_path, buffer)
    image_data = base64.standard_b64encode(buffer.getvalue()).decode("utf-8")

    caption = message.caption or "Опиши что на картинке"

    try:
        response = claude.messages.create(
            model=user_models.get(message.from_user.id, DEFAULT_MODEL),
            max_tokens=1024,
            system="Ты полезный ассистент. Отвечай на русском языке, если пользователь пишет по-русски.",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_data}},
                    {"type": "text", "text": caption}
                ]
            }]
        )
        await message.answer(response.content[0].text)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")


@dp.message()
async def handle_message(message: Message):
    if not is_allowed(message.from_user.id):
        await message.answer("Доступ запрещён")
        return

    user_id = message.from_user.id
    if user_id not in chat_histories:
        chat_histories[user_id] = []

    chat_histories[user_id].append({"role": "user", "content": message.text})

    if len(chat_histories[user_id]) > 20:
        chat_histories[user_id] = chat_histories[user_id][-20:]

    await message.bot.send_chat_action(message.chat.id, "typing")

    try:
        response = claude.messages.create(
            model=user_models.get(user_id, DEFAULT_MODEL),
            max_tokens=1024,
            system="Ты полезный ассистент. Отвечай на русском языке, если пользователь пишет по-русски.",
            messages=chat_histories[user_id]
        )
        reply = response.content[0].text
        chat_histories[user_id].append({"role": "assistant", "content": reply})
        save_history()
        await message.answer(reply)
    except Exception as e:
        await message.answer(f"Произошла ошибка: {e}")


async def main():
    load_history()
    session = AiohttpSession(proxy=PROXY_URL) if PROXY_URL else None
    bot = Bot(token=BOT_TOKEN, session=session)

    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
