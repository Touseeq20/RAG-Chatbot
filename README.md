# 📚 Document RAG Chatbot

An AI-powered **Retrieval-Augmented Generation (RAG) chatbot** built with **Streamlit, LangChain, FAISS, HuggingFace embeddings, and OpenAI (via OpenRouter API)**.  
Upload PDFs, DOCX, or TXT files, and then ask questions about your documents with **real-time streaming answers** and a **typing dots animation**.

---

## 🚀 Features
- 📂 Upload multiple PDFs, DOCX, and TXT files  
- 🔎 Automatic document chunking + embeddings with FAISS  
- 💬 Chat with your documents (chat history included)  
- ✨ Streaming answers with typing dots animation  
- 🧠 RAG-based intelligent answers  

---

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/document-rag-chatbot.git
   cd document-rag-chatbot
Create virtual environment & install dependencies:

bash
Copy code
pip install -r requirements.txt
Set your OpenRouter API Key as environment variable:

bash
Copy code
export OPENROUTER_API_KEY="your_api_key_here"   # Linux/Mac
setx OPENROUTER_API_KEY "your_api_key_here"    # Windows
▶️ Usage
Run the app:

bash
Copy code
streamlit run app.py
Then open the link shown in terminal (usually http://localhost:8501).

🛠️ Tech Stack
Frontend/UI → Streamlit

Vector Database → FAISS

Embeddings → HuggingFace (MiniLM L6 v2)

LLM → OpenRouter (OpenAI models)

Framework → LangChain
