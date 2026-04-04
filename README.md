# 🎥 YouTube RAG Chatbot

An AI-powered chatbot that lets you ask questions about any YouTube video using its transcript. Built with LangChain, FAISS, Groq LLM, and Gradio.

## ✨ Features

- 🔗 **Multi-URL support** — Load multiple YouTube videos at once and ask questions across all of them
- 💬 **Chat memory** — Remembers conversation history for natural follow-up questions
- ⚡ **Fast retrieval** — Uses FAISS vector store for efficient semantic search
- 🧠 **Powered by Groq** — Uses `llama-3.1-8b-instant` for fast, accurate answers
- 💾 **Index caching** — Saves FAISS index per video so repeated queries are instant

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| LLM | Groq (`llama-3.1-8b-instant`) |
| Embeddings | HuggingFace (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Framework | LangChain |
| Transcript | youtube-transcript-api |
| UI | Gradio |

---

## 📦 Installation

```bash
# 1. Clone the repo
git clone https://github.com/RosyPaul/Youtube-rag-chatbot.git
cd Youtube-rag-chatbot

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
cp .env.example .env
# Edit .env and add your API keys
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key

```

- Get your free Groq API key at [console.groq.com](https://console.groq.com)

---

## ▶️ Usage

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

1. **Paste one or more YouTube URLs** in the input box
2. Click **Process Video(s)** and wait for the ✅ confirmation
3. **Ask any question** about the video(s) in the chat
4. Follow-up questions work naturally thanks to chat memory!

---

## 🏗️ Project Structure

```
Youtube-rag-chatbot/
├── app.py              # Gradio UI and chat logic
├── rag_pipeline.py     # Core RAG pipeline (transcript, embeddings, LLM)
├── requirements.txt    # Python dependencies
├── .env.example        # Example environment variables
└── README.md
```

---

## 🔄 How It Works

```
YouTube URL → Fetch Transcript → Split into Chunks
     → Generate Embeddings → Store in FAISS
          → User Question → Retrieve Top-K Chunks
               → Pass to LLM with Chat History → Answer
```

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License
