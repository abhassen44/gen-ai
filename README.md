# 🤖 Gen-AI

A modular framework for building and running AI-powered agents capable of answering questions, conducting interviews, generating text, and interacting with documents — powered by Gemini Flash and other LLM APIs.

![License](https://img.shields.io/github/license/abhassen44/gen-ai)
![Last Commit](https://img.shields.io/github/last-commit/abhassen44/gen-ai)
![Issues](https://img.shields.io/github/issues/abhassen44/gen-ai)

---

## ✨ Highlights

* 🔌 **Modular Agent System** – Plug-and-play architecture for building agents like Q&A bots, document analyzers, and interviewers.
* 📄 **Resume & Text Parsing** – Tools to extract structured data from text and PDF documents.
* 🧠 **LLM Integration** – Uses Gemini Flash 2.0 (via Vertex AI) and supports multiple model backends.
* 📋 **Prompt Management** – Centralized and reusable prompt templates.
* 📊 **Custom Scorecards** – For interview evaluations and automated feedback.

---

## 📁 Project Structure

```
gen-ai/
|--whether_agent.py
|--voice_call_cursor.py
|--rag_rank_fusion.py
|--rag_decomposition_model.py
|--fine-tune.py
|--cursor_by_graph.py
```
---

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/abhassen44/gen-ai.git
cd gen-ai
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key
```

### 4. Run an Agent

```bash
python main.py
```

Edit `main.py` to switch between agents or customize their behavior.

---

## 🧪 Example Use

```python
from agents.interview import InterviewAgent

agent = InterviewAgent()
agent.run(resume_path="sample_resume.pdf")
```

---

## 🧠 Powered By

* [Gemini Flash 2.0](https://cloud.google.com/vertex-ai/docs/generative-ai/model-reference/gemini) via Vertex AI
* Python 3.10+
* OpenAI-compatible interfaces (optional)

---

## 🚣️ Roadmap

* [ ] CLI and Web Interface
* [ ] VectorDB Integration (e.g. Chroma, Pinecone)
* [ ] PDF + Image OCR input
* [ ] Persistent conversation memory
* [ ] Plugin ecosystem for new agents

---

## 🤝 Contributing

We welcome contributions!

1. Fork this repo
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Push and open a PR

---

## 🌟 Support

If you like this project, please ⭐ the repo and share it with others!