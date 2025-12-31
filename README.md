# 🌙 AI-Based Telegram Bot (Poppy AI Companion)

An **emotionally intelligent, Poppy-style AI companion** built on Telegram that feels less like a bot and more like a **calm, supportive friend**.

This project combines **local LLMs**, **vector-based long-term memory**, **mood detection**, and **adaptive personality learning** to deliver human-like conversations — without relying on paid APIs.

---

## ✨ Key Features

### 🤖 Conversational AI (Poppy-Style)
- Calm, dreamy, emotionally aware responses
- Talks like a close friend, not a chatbot
- No robotic or “AI assistant” tone

### 🧠 Persistent Long-Term Memory
- FAISS vector database for semantic memory
- Remembers conversations across restarts
- Recalls relevant past interactions contextually

### 🌙 Mood Detection & Emotional Intelligence
- Detects user emotions (sad, happy, anxious, calm, lonely, etc.)
- Adjusts tone and responses empathetically
- Adds emotion-aware emojis for human touch

### 🧬 Adaptive Personality Learning
- Learns how the user prefers to be spoken to
- Becomes softer, cheerful, or balanced over time
- Personality evolves with conversation history

### 🖤 Night / Day Personality Modes
- 🌙 **Night Mode**: calm, dreamy, emotionally deep
- ☀️ **Day Mode**: friendly, warm, conversational
- Switch instantly using inline Telegram buttons

### 🎭 Human-like Experience
- Typing delay to simulate real conversation
- Emotion-based emojis
- Short, natural replies

### ☁️ Cloud Deployable (24/7)
- Railway / Render compatible
- Environment-variable based configuration
- No paid APIs required

---

## 🧠 Tech Stack

| Component | Technology |
|---------|------------|
| Language | Python 3.10+ |
| Bot Framework | python-telegram-bot (v20+) |
| LLM | Ollama (Mistral – local) |
| Memory | FAISS (Vector Database) |
| Embeddings | Sentence-Transformers |
| Deployment | Railway / Render |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Unknowncoder3/Ai-Based-TelegramBot.git
cd Ai-Based-TelegramBot
````

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Install & Run Ollama

```bash
ollama pull mistral
ollama serve
```

### 4️⃣ Set Environment Variable

```bash
export TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
```

### 5️⃣ Run the Bot

```bash
python main.py
```

---

## 💬 Example Interaction

**User:**

> I feel lonely tonight

**poppy:**

> You’re not alone right now… I’m here with you 🌙🤍

---

## 📸 Demo

> *(Add a screen-recorded demo GIF here for maximum impact)*

```markdown
![Lucid AI Bot Demo](demo.gif)
```

---

## 🎯 Use Cases

* AI companion / emotional support chatbot
* Conversational AI research
* Telegram bot development showcase
* GenAI portfolio project
* Interview-ready system design example

---

## 🧠 What Makes This Project Stand Out

* Uses **local LLMs** (privacy-first, cost-free)
* Implements **real long-term memory**, not chat history
* Emotion-aware, personality-driven responses
* Production-ready architecture
* Clear separation of AI, memory, and bot logic

---

## 👤 Author

**Snehasish Das**
Final Year CSBS Student | GenAI & Applied AI Developer

* GitHub: [https://github.com/Unknowncoder3](https://github.com/Unknowncoder3)
* LinkedIn: *(add if you want)*

---

## ⭐ If you like this project

Give it a ⭐ and feel free to fork, improve, or build your own Lucid companion 🌙


