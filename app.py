import warnings
warnings.filterwarnings(“ignore”)
import os
os.environ[“TOKENIZERS_PARALLELISM”] = “false”

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(
page_title=“Atlas Intelligence”,
page_icon=“🏗️”,
layout=“centered”,
initial_sidebar_state=“collapsed”,
)

st.markdown(”””

<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Karla:wght@300;400;500&display=swap');

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stHeader"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { display: none !important; }

/* ── Base ── */
html, body, .stApp {
    font-family: 'Karla', sans-serif !important;
    background-color: #F5F3EE !important;
    color: #1a1a18 !important;
}

.block-container {
    max-width: 780px !important;
    padding-top: 2rem !important;
    padding-bottom: 6rem !important;
    margin: 0 auto !important;
}

/* ── Header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 20px 0;
    border-bottom: 2px solid #1a1a18;
    margin-bottom: 28px;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: #1a1a18;
    margin: 0;
}
.app-subtitle {
    font-size: 0.78rem;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 2px 0 0 0;
}
.header-badge {
    background: #1a1a18;
    color: #F5F3EE;
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 5px 13px;
    border-radius: 20px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 48px 20px 24px;
}
.empty-state-icon { font-size: 2.5rem; display: block; margin-bottom: 12px; }
.empty-state-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.05rem;
    font-weight: 700;
    color: #888;
    margin-bottom: 6px;
}
.empty-state-sub { font-size: 0.85rem; color: #bbb; }

/* ── Suggestion buttons ── */
.stButton > button {
    background: #fff !important;
    color: #444 !important;
    border: 1.5px solid #dedad3 !important;
    border-radius: 20px !important;
    font-family: 'Karla', sans-serif !important;
    font-size: 0.83rem !important;
    padding: 8px 14px !important;
    width: 100% !important;
    transition: all 0.15s ease !important;
}
.stButton > button:hover {
    border-color: #1a1a18 !important;
    color: #1a1a18 !important;
    background: #f0ede8 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 6px 0 !important;
    animation: fadeUp 0.25s ease;
}
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Bot bubble */
[data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    color: #1a1a18 !important;
    border: 1.5px solid #e0ddd6 !important;
    border-radius: 4px 16px 16px 16px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 8px rgba(0,0,0,0.06) !important;
    font-family: 'Karla', sans-serif !important;
    font-size: 0.92rem !important;
}

/* All text in bubbles */
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] td,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] strong,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] strong {
    color: #1a1a18 !important;
}

/* Tables */
[data-testid="stChatMessageContent"] table,
[data-testid="stMarkdownContainer"] table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.87rem;
    margin: 10px 0;
}
[data-testid="stChatMessageContent"] thead tr,
[data-testid="stMarkdownContainer"] thead tr { background: #1a1a18 !important; }
[data-testid="stChatMessageContent"] thead th,
[data-testid="stChatMessageContent"] thead th p,
[data-testid="stMarkdownContainer"] thead th,
[data-testid="stMarkdownContainer"] thead th p {
    color: #fff !important;
    background: transparent !important;
    padding: 9px 14px !important;
    font-weight: 700 !important;
    font-size: 0.81rem !important;
    white-space: nowrap !important;
    text-align: left !important;
}
[data-testid="stChatMessageContent"] tbody tr:nth-child(even),
[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) { background: #f8f7f4 !important; }
[data-testid="stChatMessageContent"] tbody tr:nth-child(odd),
[data-testid="stMarkdownContainer"] tbody tr:nth-child(odd)  { background: #fff !important; }
[data-testid="stChatMessageContent"] td,
[data-testid="stMarkdownContainer"] td {
    padding: 8px 14px !important;
    border-bottom: 1px solid #ece9e2 !important;
    color: #1a1a18 !important;
}

/* Divider */
.date-divider {
    display: flex; align-items: center; gap: 12px;
    margin: 20px 0 16px; color: #ccc;
    font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase;
}
.date-divider::before, .date-divider::after {
    content: ''; flex: 1; height: 1px; background: #e5e2db;
}

/* ── Input form ── */
.stForm {
    background: #F5F3EE !important;
    border-top: 1.5px solid #e0ddd6 !important;
    padding-top: 12px !important;
}
.stTextInput > div > div > input {
    background: #fff !important;
    color: #1a1a18 !important;
    border: 1.5px solid #d0cdc6 !important;
    border-radius: 12px !important;
    font-family: 'Karla', sans-serif !important;
    font-size: 0.92rem !important;
    padding: 11px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a1a18 !important;
    box-shadow: 0 0 0 2px rgba(26,26,24,0.08) !important;
}
.stTextInput > div > div > input::placeholder { color: #b5b2ab !important; }

.stFormSubmitButton > button {
    background: #1a1a18 !important;
    color: #F5F3EE !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    padding: 11px 20px !important;
    height: 46px !important;
    white-space: nowrap !important;
    transition: background 0.15s ease !important;
}
.stFormSubmitButton > button:hover { background: #333 !important; }

/* Clear button */
.clear-btn > .stButton > button {
    background: transparent !important;
    color: #bbb !important;
    border: 1px solid #e0ddd6 !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    width: auto !important;
}
</style>

“””, unsafe_allow_html=True)

# ── Load chain ────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=“⏳ Chargement…”)
def load_chain():
MISTRAL_API_KEY = os.environ.get(“MISTRAL_API_KEY”, “”)
if not MISTRAL_API_KEY:
try:
MISTRAL_API_KEY = st.secrets[“MISTRAL_API_KEY”]
except (FileNotFoundError, KeyError):
st.error(“⚠️ Clé API Mistral introuvable.”)
st.stop()

```
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

if not os.path.exists("faiss_index"):
    st.error("⚠️ Index FAISS introuvable. Lancez `python ingest.py` d'abord.")
    st.stop()

db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 15})

llm = ChatMistralAI(
    mistral_api_key=MISTRAL_API_KEY,
    model="mistral-medium-latest",
    temperature=0.1,
)

prompt = PromptTemplate(
    template="""Tu es un assistant RH et business intelligent pour des entreprises algériennes de travaux publics.
```

RÈGLES IMPORTANTES :

- Utilise UNIQUEMENT les données du contexte ci-dessous.
- Cherche dans TOUTES les sections du contexte avant de répondre.
- Si la question porte sur les dépenses (gasoil, graisse, huile, etc.), cherche dans DÉPENSES OPÉRATIONNELLES.
- Si la question porte sur les charges (CNAS, CACOBATPH, IRG), cherche dans CHARGES SOCIALES et TOTAL CHARGES.
- Ne dis JAMAIS qu’une information est absente si elle apparaît dans le contexte.
- Réponds en français, de façon claire et structurée.
- Pour les listes d’employés ou de dépenses, utilise TOUJOURS un tableau Markdown.
- Sois direct et concis.

Contexte :
{context}

Question : {question}

Réponse :”””,
input_variables=[“context”, “question”],
)

```
def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
return chain
```

chain = load_chain()

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown(”””

<div class="app-header">
    <div>
        <p class="app-title">🏗️ Atlas Intelligence</p>
        <p class="app-subtitle">RH & Gestion d'entreprise</p>
    </div>
    <span class="header-badge">Mistral AI</span>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────

if “messages” not in st.session_state:
st.session_state.messages = []

# ── Empty state + suggestion chips ───────────────────────────────────────────

if not st.session_state.messages:
st.markdown(”””
<div class="empty-state">
<span class="empty-state-icon">🏗️</span>
<p class="empty-state-text">Comment puis-je vous aider ?</p>
<p class="empty-state-sub">Posez une question sur vos employés, salaires ou dépenses</p>
</div>
“””, unsafe_allow_html=True)

```
suggestions = [
    "👥 Employés de l'Atlas Machinery",
    "💰 Total salaires Noor Location",
    "⛽ Dépenses gasoil des deux companies",
    "📊 Charges mensuelles totales",
]
cols = st.columns(2)
for i, s in enumerate(suggestions):
    with cols[i % 2]:
        if st.button(s, key=f"chip_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": s})
            with st.spinner("Réflexion..."):
                answer = chain.invoke(s)
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
```

# ── Chat history ──────────────────────────────────────────────────────────────

if st.session_state.messages:
st.markdown(’<div class="date-divider">Conversation</div>’, unsafe_allow_html=True)

for msg in st.session_state.messages:
with st.chat_message(msg[“role”], avatar=“🧑” if msg[“role”] == “user” else “🤖”):
st.markdown(msg[“content”])

# ── Clear button ──────────────────────────────────────────────────────────────

if st.session_state.messages:
st.markdown(’<div class="clear-btn">’, unsafe_allow_html=True)
if st.button(“✕ Effacer la conversation”):
st.session_state.messages = []
st.rerun()
st.markdown(’</div>’, unsafe_allow_html=True)

# ── Input form ────────────────────────────────────────────────────────────────

st.markdown(”—”)
with st.form(“chat_form”, clear_on_submit=True):
col1, col2 = st.columns([6, 1])
with col1:
user_input = st.text_input(
“msg”,
placeholder=“Ex : Quels sont les employés de l’Atlas Machinery ?”,
label_visibility=“collapsed”,
)
with col2:
submitted = st.form_submit_button(“Envoyer”)

if submitted and user_input.strip():
st.session_state.messages.append({“role”: “user”, “content”: user_input})
with st.spinner(“Réflexion…”):
answer = chain.invoke(user_input)
st.session_state.messages.append({“role”: “assistant”, “content”: answer})
st.rerun()
