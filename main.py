"""
poppy AI COMPANION – FINAL VERSION
Features:
- Persistent long-term memory (FAISS + disk)
- Mood detection
- Personality learning
- Night / Day mode
- Inline buttons
- Emotion emojis + typing delay
"""

import os
import pickle
import asyncio
import logging
from datetime import datetime

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# ---------------- LOGGING ---------------- #
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------- CONFIG ---------------- #
TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")  # use env var for cloud
MEMORY_DIR = "memory"
FAISS_PATH = f"{MEMORY_DIR}/faiss.index"
TEXT_PATH = f"{MEMORY_DIR}/memory.pkl"

os.makedirs(MEMORY_DIR, exist_ok=True)

# ---------------- AI MODEL ---------------- #
llm = OllamaLLM(model="mistral")

# ---------------- EMBEDDINGS ---------------- #
embedder = SentenceTransformer("all-MiniLM-L6-v2")
EMBED_DIM = 384

# ---------------- LOAD MEMORY ---------------- #
if os.path.exists(FAISS_PATH):
    index = faiss.read_index(FAISS_PATH)
    with open(TEXT_PATH, "rb") as f:
        memory_texts = pickle.load(f)
else:
    index = faiss.IndexFlatL2(EMBED_DIM)
    memory_texts = []

# ---------------- USER STATE ---------------- #
user_state = {}      # {user_id: {"mode": "day"/"night", "mood": "neutral"}}
user_profile = {}    # {user_id: {"tone": "soft"/"cheerful"/"balanced"}}

# ---------------- PROMPTS ---------------- #
DAY_PROMPT = """
You are poppy — a calm, kind, emotionally intelligent friend.
You speak gently, warmly, and naturally.
You keep replies short and human.
"""

NIGHT_PROMPT = """
You are poppy — a quiet, dreamy, comforting presence.
Your words feel like moonlight and midnight thoughts.
You speak softly and emotionally.
"""

# ---------------- EMOJI MAP ---------------- #
EMOJI_MAP = {
    "sad": "🤍",
    "happy": "✨",
    "angry": "🌧️",
    "anxious": "🌊",
    "calm": "🌿",
    "lonely": "🫂",
    "neutral": "🌙",
}

# ---------------- UTILS ---------------- #
def detect_mood(text: str):
    t = text.lower()
    if any(w in t for w in ["sad", "tired", "lonely", "depressed", "broken"]):
        return "sad"
    if any(w in t for w in ["happy", "excited", "great", "good"]):
        return "happy"
    if any(w in t for w in ["angry", "mad", "frustrated"]):
        return "angry"
    if any(w in t for w in ["anxious", "nervous", "worried"]):
        return "anxious"
    if any(w in t for w in ["calm", "relaxed", "peaceful"]):
        return "calm"
    if any(w in t for w in ["alone", "isolated"]):
        return "lonely"
    return "neutral"


def persist_memory():
    faiss.write_index(index, FAISS_PATH)
    with open(TEXT_PATH, "wb") as f:
        pickle.dump(memory_texts, f)


def store_memory(text: str):
    emb = embedder.encode([text])
    index.add(np.array(emb))
    memory_texts.append(text)
    persist_memory()


def recall_memory(query: str, k=2):
    if index.ntotal == 0:
        return ""
    q_emb = embedder.encode([query])
    _, idxs = index.search(np.array(q_emb), k)
    return "\n".join(memory_texts[i] for i in idxs[0] if i < len(memory_texts))


# ---------------- COMMANDS ---------------- #
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_state[user_id] = {"mode": "day", "mood": "neutral"}
    user_profile[user_id] = {"tone": "balanced"}

    keyboard = [
        [
            InlineKeyboardButton("🌙 Night Mode", callback_data="night"),
            InlineKeyboardButton("☀️ Day Mode", callback_data="day"),
        ]
    ]

    await update.message.reply_text(
        "Hey… I’m poppy 🌙\nI remember things, sense moods, and stay with you.\nHow are you feeling?",
        reply_markup=InlineKeyboardMarkup(keyboard),
    )


# ---------------- BUTTON HANDLER ---------------- #
async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    user_id = query.from_user.id
    user_state.setdefault(user_id, {"mode": "day", "mood": "neutral"})

    if query.data == "night":
        user_state[user_id]["mode"] = "night"
        await query.edit_message_text("🌙 Night mode on… I’m right here.")
    elif query.data == "day":
        user_state[user_id]["mode"] = "day"
        await query.edit_message_text("☀️ Day mode on. Let’s talk freely.")


# ---------------- CHAT HANDLER ---------------- #
async def chat(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_msg = update.message.text

    user_state.setdefault(user_id, {"mode": "day", "mood": "neutral"})
    user_profile.setdefault(user_id, {"tone": "balanced"})

    mood = detect_mood(user_msg)
    user_state[user_id]["mood"] = mood

    # personality learning
    if mood == "sad":
        user_profile[user_id]["tone"] = "soft"
    elif mood == "happy":
        user_profile[user_id]["tone"] = "cheerful"

    store_memory(f"{datetime.now()}: {user_msg}")
    recalled = recall_memory(user_msg)

    system_prompt = NIGHT_PROMPT if user_state[user_id]["mode"] == "night" else DAY_PROMPT
    tone = user_profile[user_id]["tone"]

    prompt = f"""
{system_prompt}

Adapt your tone to be {tone}.
User mood: {mood}

Relevant memories:
{recalled}

User:
{user_msg}

Lucid:
"""

    await context.bot.send_chat_action(
        chat_id=update.effective_chat.id,
        action="typing"
    )
    await asyncio.sleep(1.5)

    try:
        response = llm.invoke(prompt)
    except Exception:
        response = "I’m still here… even if words feel heavy 🤍"

    emoji = EMOJI_MAP.get(mood, "🌙")
    await update.message.reply_text(f"{response} {emoji}")


# ---------------- MAIN ---------------- #
def main():
    if not TOKEN:
        raise ValueError("TELEGRAM_BOT_TOKEN not set")

    app = ApplicationBuilder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(button_handler))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, chat))

    print("🌙 Lucid AI Companion is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
