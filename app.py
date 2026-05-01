import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

st.set_page_config(
    page_title="Resume Chatbot | Mohamed El Sayed",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #0f172a;
    color: #e5e7eb;
}
.block-container {
    padding-top: 4rem;
    max-width: 900px;
}
h1 {
    color: #f8fafc;
    font-size: 3rem;
    font-weight: 800;
}
p, label, .stMarkdown {
    color: #cbd5e1;
}
.card {
    background: #111827;
    border: 1px solid #334155;
    border-radius: 18px;
    padding: 24px;
    margin-bottom: 24px;
}
.badge {
    display: inline-block;
    background: #1e293b;
    border: 1px solid #475569;
    color: #93c5fd;
    border-radius: 999px;
    padding: 6px 12px;
    margin: 4px;
    font-size: 0.85rem;
}
.stTextInput input {
    background: #020617;
    color: #f8fafc;
    border: 1px solid #334155;
    border-radius: 12px;
}
.stButton button {
    background: #2563eb;
    color: white;
    border-radius: 12px;
    border: none;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
# Resume Chatbot

<div class="card">
Ask questions about my background, skills, cloud projects, data engineering work, and technical experience.
</div>

<span class="badge">OpenAI</span>
<span class="badge">LangChain</span>
<span class="badge">FAISS</span>
<span class="badge">Streamlit</span>
<span class="badge">AWS Ready</span>
""", unsafe_allow_html=True)

@st.cache_resource
def load_chain():
    resume_loader = PyPDFLoader("docs/resume.pdf")
    profile_loader = TextLoader("docs/profile.md", encoding="utf-8")

    docs = resume_loader.load() + profile_loader.load()

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    llm = ChatOpenAI(model="gpt-4.1-mini")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    return qa

qa = load_chain()

query = st.text_input("Ask about Mohamed:", placeholder="Example: What roles is Mohamed best suited for?")

if query:
    with st.spinner("Thinking..."):
        response = qa.invoke(query)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(response["result"])
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
---
Built with **RAG**, **LangChain**, **FAISS**, **OpenAI**, and **Streamlit**.  
Designed to match [mohamedthedev.com](https://mohamedthedev.com).
""")