import streamlit as st
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
import tempfile
import time
import threading
import os

# ---------------------------
# OpenRouter Client Setup (LLM)
# ---------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # use environment variable
)

# ---------------------------
# Build Vectorstore from Uploaded Docs
# ---------------------------
def process_uploaded_files(uploaded_files):
    docs = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
            tmp_file.write(uploaded_file.read())
            file_path = tmp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs.extend(loader.load())
        elif uploaded_file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            docs.extend(loader.load())
        elif uploaded_file.name.endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_vectorstore(documents):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore

# ---------------------------
# Streaming LLM with Typing Dots
# ---------------------------
def ask_llm_stream(query, retriever, chat_history):
    docs = retriever.get_relevant_documents(query)
    context = "\n".join([d.page_content for d in docs])

    history_text = "\n".join([f"User: {h[0]}\nAssistant: {h[1]}" for h in chat_history])

    prompt = f"""
You are a helpful assistant. Use the following context to answer the user’s questions.

Chat History:
{history_text}

Context from documents:
{context}

Question:
{query}

Answer:
"""

    full_response = ""
    msg_placeholder = st.empty()
    stop_animation = False

    # Typing dots animation
    def animate_typing():
        while not stop_animation:
            for dots in ["", ".", "..", "..."]:
                if stop_animation:
                    break
                msg_placeholder.markdown(
                    f"<div style='text-align: left; background-color:#F1F0F0; "
                    f"color:#000000; padding:8px; border-radius:10px; "
                    f"display:inline-block;'>Assistant is typing{dots}</div>",
                    unsafe_allow_html=True,
                )
                time.sleep(0.4)

    animation_thread = threading.Thread(target=animate_typing)
    animation_thread.start()

    # Stream response
    stream = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    first_chunk_received = False
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            if not first_chunk_received:
                stop_animation = True
                animation_thread.join()
                first_chunk_received = True

            full_response += delta
            msg_placeholder.markdown(
                f"<div style='text-align: left; background-color:#F1F0F0; "
                f"color:#000000; padding:8px; border-radius:10px; display:inline-block;'>{full_response}</div>",
                unsafe_allow_html=True,
            )

    stop_animation = True
    return full_response

# ---------------------------
# Streamlit UI
# ---------------------------
def main():
    st.set_page_config(page_title="📚 Document RAG Chatbot", layout="wide")
    st.title("📚 Document RAG Chatbot ")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_files = st.file_uploader(
        "📂 Upload PDF, DOCX, or TXT files",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        docs = process_uploaded_files(uploaded_files)
        st.session_state.vectorstore = build_vectorstore(docs)
        st.success("✅ Knowledge Base built automatically!")

    # Show previous chat
    for user_msg, bot_msg in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(
                f"<div style='text-align: right; background-color:#DCF8C6; "
                f"color:#000000; padding:8px; border-radius:10px; display:inline-block;'>{user_msg}</div>",
                unsafe_allow_html=True,
            )
        with st.chat_message("assistant"):
            st.markdown(
                f"<div style='text-align: left; background-color:#F1F0F0; "
                f"color:#000000; padding:8px; border-radius:10px; display:inline-block;'>{bot_msg}</div>",
                unsafe_allow_html=True,
            )

    # New queries
    if st.session_state.vectorstore:
        query = st.chat_input("💬 Ask something about your documents...")
        if query:
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})

            with st.chat_message("user"):
                st.markdown(
                    f"<div style='text-align: right; background-color:#DCF8C6; "
                    f"color:#000000; padding:8px; border-radius:10px; display:inline-block;'>{query}</div>",
                    unsafe_allow_html=True,
                )

            with st.chat_message("assistant"):
                answer = ask_llm_stream(query, retriever, st.session_state.chat_history)

            st.session_state.chat_history.append((query, answer))


if __name__ == "__main__":
    main()
