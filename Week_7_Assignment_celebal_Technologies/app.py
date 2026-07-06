import os
import tempfile

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Document Q&A (RAG)", page_icon="📄")
st.title("📄 Document Question Answering System")
st.caption("Ask questions about a document — the assistant answers only from its content.")

# ---------- API key ----------
api_key = st.secrets.get("OPENAI_API_KEY", None) if hasattr(st, "secrets") else None
if not api_key:
    api_key = st.sidebar.text_input("OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key

# ---------- Document source ----------
st.sidebar.header("Document source")
source = st.sidebar.radio(
    "Choose a document",
    ["Sample: cricket.txt", "Sample: environment.pdf", "Upload my own"],
)

uploaded_file = None
if source == "Upload my own":
    uploaded_file = st.sidebar.file_uploader("Upload a .pdf or .txt file", type=["pdf", "txt"])


@st.cache_data(show_spinner="Loading document...")
def load_text_from_path(path: str, is_pdf: bool) -> str:
    if is_pdf:
        docs = PyPDFLoader(path).load()
        return "\n\n".join(doc.page_content for doc in docs)
    docs = TextLoader(path, encoding="utf-8").load()
    return docs[0].page_content


@st.cache_data(show_spinner="Loading document...")
def load_text_from_bytes(file_bytes: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return load_text_from_path(tmp_path, is_pdf=(suffix == ".pdf"))
    finally:
        os.remove(tmp_path)


document_text = None
doc_label = None

if source == "Sample: cricket.txt":
    document_text = load_text_from_path(os.path.join(BASE_DIR, "cricket.txt"), is_pdf=False)
    doc_label = "cricket.txt"
elif source == "Sample: environment.pdf":
    document_text = load_text_from_path(os.path.join(BASE_DIR, "environment.pdf"), is_pdf=True)
    doc_label = "environment.pdf"
elif uploaded_file is not None:
    suffix = ".pdf" if uploaded_file.name.lower().endswith(".pdf") else ".txt"
    document_text = load_text_from_bytes(uploaded_file.getvalue(), suffix)
    doc_label = uploaded_file.name

# ---------- Reset chat when the document changes ----------
if "doc_label" not in st.session_state or st.session_state.doc_label != doc_label:
    st.session_state.doc_label = doc_label
    st.session_state.messages = []  # what we render in the UI

if not api_key:
    st.info("Enter your OpenAI API key in the sidebar to start chatting.")
    st.stop()

if not document_text:
    st.info("Upload a document (or pick a sample) in the sidebar to start chatting.")
    st.stop()

st.success(f"Loaded: {doc_label}")

# ---------- Render existing chat ----------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------- Chat input ----------
user_input = st.chat_input("Ask a question about the document...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    system_prompt = SystemMessage(content=f"""You are a helpful assistant. Answer the user's questions
only based on the following text. If the answer is not in the text, say you don't know.

Text:
{document_text}
""")

    lc_history = [system_prompt]
    for msg in st.session_state.messages[:-1]:
        if msg["role"] == "user":
            lc_history.append(HumanMessage(content=msg["content"]))
        else:
            lc_history.append(AIMessage(content=msg["content"]))
    lc_history.append(HumanMessage(content=user_input))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                model = ChatOpenAI()
                result = model.invoke(lc_history)
                answer = result.content
            except Exception as e:
                answer = f"Error calling OpenAI: {e}"
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
