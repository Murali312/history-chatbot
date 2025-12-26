# 📜 HistoryBot: Historical Figures Chatbot

HistoryBot is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions about historical figures based on a custom PDF knowledge base. It uses **LangChain**, **Ollama** (local LLMs), **ChromaDB**, and **Gradio** for the user interface.

## 🚀 Features

* **RAG Pipeline**: Loads a PDF, chunks text, and retrieves relevant context for accurate answers.
* **Local LLM Support**: Runs entirely locally using **Ollama** (Llama3 for generation, Granite for embeddings).
* **Vector Search**: Uses **ChromaDB** for efficient similarity search.
* **Memory**: Tracks conversation history to handle follow-up questions.
* **Tracing**: Integrated with **LangSmith** for debugging and monitoring LLM traces.
* **User Interface**: Clean web interface built with **Gradio**.

---

## 🛠️ Prerequisites

Before you begin, ensure you have the following installed:

1.  **Python 3.10+**
2.  **[Ollama](https://ollama.com/)** (Running in the background)
3.  **LangSmith API Key** (Optional, for tracing)

### 📥 Download Ollama Models
You must pull the specific models used in the code before running the bot:

```bash
ollama pull llama3.2:3b
ollama pull granite-embedding:latest
