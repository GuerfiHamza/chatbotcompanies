import warnings
warnings.filterwarnings("ignore")

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Atlas Intelligence",
    page_icon="🏗️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Karla:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"], .stApp {
    font-family: 'Karla', sans-serif;
    background: #F5F3EE !important;
    color: #1a1a18;
}

/* Hide ALL Streamlit chrome — works on Cloud too */
[data-testid="collapsedControl"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }
[data-testid="stDecoration"] { display: none !important; }
[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
header { display: none !important; }
.stDeployButton { display: none !important; }
.stAppDeployButton { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }

/* ── Main container ── */
.block-container {
    max-width: 780px !important;
    padding: 0 24px 120px 24px !important;
    margin: 0 auto !important;
}

/* ── App header ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 32px 0 20px 0;
    border-bottom: 2px solid #1a1a18;
    margin-bottom: 32px;
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #1a1a18;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-subtitle {
    font-size: 0.8rem;
    color: #888;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin: 0;
}
.header-badge {
    background: #1a1a18;
    color: #F5F3EE;
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 6px 14px;
    border-radius: 20px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 60px 20px;
    color: #aaa;
}
.empty-state-icon {
    font-size: 3rem;
    margin-bottom: 16px;
    display: block;
    opacity: 0.4;
}
.empty-state-text {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 600;
    color: #bbb;
    margin-bottom: 8px;
}
.empty-state-sub {
    font-size: 0.85rem;
    color: #ccc;
}

/* ── Native st.chat_message styling ── */
@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* All chat messages */
[data-testid="stChatMessage"] {
    animation: fadeUp 0.3s ease;
    background: transparent !important;
    padding: 4px 0 !important;
}

/* User bubble — target the inner content div */
[data-testid="stChatMessage"].st-emotion-cache-janbn0,
[data-testid="stChatMessage"]:has(> div > [data-testid="chatAvatarIcon-user"]) {
    flex-direction: row-reverse;
}

/* Bot message content wrapper */
[data-testid="stChatMessage"] .stChatMessageContent,
[data-testid="stChatMessage"] [class*="stChatMessageContent"] {
    border-radius: 4px 18px 18px 18px;
    border: 1.5px solid #e0ddd6;
    background: #ffffff !important;
    color: #1a1a18 !important;
    padding: 14px 18px;
    font-family: 'Karla', sans-serif;
    font-size: 0.93rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    overflow-x: auto;
}

/* Use st.chat_message name attribute to distinguish user vs assistant */
[data-testid="stChatMessage"]:nth-child(odd) [class*="stChatMessageContent"],
[data-testid="stChatMessage"]:nth-child(odd) [data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    color: #1a1a18 !important;
    border: 1.5px solid #e0ddd6 !important;
    border-radius: 4px 18px 18px 18px !important;
}

/* Force ALL text inside any chat bubble to be readable */
[data-testid="stChatMessageContent"] {
    background: #ffffff !important;
    color: #1a1a18 !important;
    border: 1.5px solid #e0ddd6 !important;
    border-radius: 12px !important;
    padding: 14px 18px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06) !important;
}
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] td,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] strong,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] span,
[data-testid="stMarkdownContainer"] strong {
    color: #1a1a18 !important;
}

/* Table headers — white text on dark bg */
[data-testid="stChatMessageContent"] thead tr,
[data-testid="stMarkdownContainer"] thead tr {
    background: #1a1a18 !important;
}
[data-testid="stChatMessageContent"] thead th,
[data-testid="stChatMessageContent"] thead th p,
[data-testid="stChatMessageContent"] thead th span,
[data-testid="stMarkdownContainer"] thead th,
[data-testid="stMarkdownContainer"] thead th p {
    color: #ffffff !important;
    background: transparent !important;
    padding: 9px 14px !important;
    font-weight: 700 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.04em !important;
    white-space: nowrap !important;
}
[data-testid="stChatMessageContent"] tbody tr:nth-child(even),
[data-testid="stMarkdownContainer"] tbody tr:nth-child(even) { background: #f8f7f4 !important; }
[data-testid="stChatMessageContent"] tbody tr:nth-child(odd),
[data-testid="stMarkdownContainer"] tbody tr:nth-child(odd)  { background: #fff !important; }
[data-testid="stChatMessageContent"] td,
[data-testid="stMarkdownContainer"] td {
    padding: 8px 14px !important;
    border-bottom: 1px solid #ece9e2 !important;
    white-space: nowrap !important;
    color: #1a1a18 !important;
}

/* Avatar styling */
[data-testid="chatAvatarIcon-user"] {
    background: #e0ddd6 !important;
    color: #1a1a18 !important;
}
[data-testid="chatAvatarIcon-assistant"] {
    background: #1a1a18 !important;
    color: #F5F3EE !important;
}

/* ── Divider between sessions ── */
.date-divider {
    display: flex;
    align-items: center;
    gap: 12px;
    margin: 28px 0 20px;
    color: #bbb;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.date-divider::before, .date-divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: #e0ddd6;
}

/* ── Input area (sticky bottom) ── */
.stForm {
    position: fixed !important;
    bottom: 0 !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: 100% !important;
    max-width: 780px !important;
    background: #F5F3EE !important;
    border-top: 1.5px solid #e0ddd6 !important;
    padding: 16px 24px 20px !important;
    z-index: 999 !important;
}

.stTextInput > div > div > input {
    background: #fff !important;
    color: #1a1a18 !important;
    border: 1.5px solid #d0cdc6 !important;
    border-radius: 12px !important;
    font-family: 'Karla', sans-serif !important;
    font-size: 0.93rem !important;
    padding: 12px 16px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus {
    border-color: #1a1a18 !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.1) !important;
}
.stTextInput > div > div > input::placeholder {
    color: #b0ada6 !important;
}

.stFormSubmitButton > button {
    background: #1a1a18 !important;
    color: #F5F3EE !important;
    border: none !important;
    border-radius: 12px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em !important;
    padding: 12px 20px !important;
    height: 48px !important;
    transition: background 0.2s ease, transform 0.1s ease !important;
    white-space: nowrap !important;
}
.stFormSubmitButton > button:hover {
    background: #333 !important;
    transform: translateY(-1px) !important;
}
.stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

/* ── Suggestion chip buttons ── */
.stButton > button {
    background: #fff !important;
    color: #444 !important;
    border: 1.5px solid #e0ddd6 !important;
    border-radius: 20px !important;
    font-family: 'Karla', sans-serif !important;
    font-size: 0.82rem !important;
    padding: 8px 16px !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04) !important;
}
.stButton > button:hover {
    border-color: #1a1a18 !important;
    color: #1a1a18 !important;
    background: #f5f3ee !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08) !important;
    transform: translateY(-1px) !important;
}

/* ── Clear button (last button on page gets distinct style) ── */
div[data-testid="stButton"]:last-of-type > button {
    background: transparent !important;
    color: #bbb !important;
    border: 1px solid #e8e5df !important;
    border-radius: 8px !important;
    font-size: 0.78rem !important;
    padding: 4px 12px !important;
    box-shadow: none !important;
}
div[data-testid="stButton"]:last-of-type > button:hover {
    color: #e05555 !important;
    border-color: #e05555 !important;
    transform: none !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #1a1a18 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load chain (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Chargement de la base de connaissances...")
def load_chain():
    MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY", "")
    if not MISTRAL_API_KEY:
        try:
            MISTRAL_API_KEY = st.secrets["MISTRAL_API_KEY"]
        except (FileNotFoundError, KeyError):
            st.error("⚠️ Clé API Mistral introuvable. Ajoutez MISTRAL_API_KEY dans votre fichier .env")
            st.stop()

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

RÈGLES IMPORTANTES :
- Utilise UNIQUEMENT les données du contexte ci-dessous.
- Le contexte contient plusieurs sections par entreprise : EMPLOYÉS, SALAIRES, CHARGES SOCIALES, DÉPENSES OPÉRATIONNELLES, TOTAL CHARGES.
- Cherche dans TOUTES les sections du contexte avant de répondre.
- Si la question porte sur les dépenses (gasoil, graisse, huile, etc.), cherche dans la section DÉPENSES OPÉRATIONNELLES.
- Si la question porte sur les charges (CNAS, CACOBATPH, IRG), cherche dans CHARGES SOCIALES et TOTAL CHARGES.
- Ne dis JAMAIS qu'une information est absente si elle apparaît quelque part dans le contexte.
- Réponds toujours en français, de façon claire et structurée.
- Pour les listes d'employés ou de dépenses, utilise TOUJOURS un tableau Markdown (| Col | Col |).
- Pour les listes simples, utilise des tirets (-).
- N'ajoute pas de lignes vides inutiles entre les éléments.
- Sois direct et concis, sans phrases de remplissage.

Contexte complet :
{context}

Question : {question}

Réponse :""",
        input_variables=["context", "question"],
    )

    def format_docs(docs):
        # Join all retrieved chunks with clear separators
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

chain = load_chain()

# ── Header (rendered after chain loads) ──────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div>
        <p class="app-title">🏗️ Atlas Intelligence</p>
        <p class="app-subtitle">RH & Gestion d\'entreprise</p>
    </div>
    <span class="header-badge">Mistral AI</span>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="empty-state">
        <span class="empty-state-icon">🏗️</span>
        <p class="empty-state-text">Comment puis-je vous aider ?</p>
        <p class="empty-state-sub">Posez une question sur vos employés, salaires ou dépenses</p>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "👥 Employés de l'Atlas Machinery",
        "💰 Total salaires Noor Location",
        "⛽ Dépenses gasoil des deux companies",
        "📊 Charges mensuelles totales",
    ]

    cols = st.columns(2)
    for i, suggestion in enumerate(suggestions):
        with cols[i % 2]:
            if st.button(suggestion, key=f"chip_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": suggestion})
                with st.spinner(""):
                    answer = chain.invoke(suggestion)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
if st.session_state.messages:
    st.markdown('<div class="date-divider">Conversation</div>', unsafe_allow_html=True)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

# Clear button (only shown when there are messages)
if st.session_state.messages:
    if st.button("✕ Effacer la conversation"):
        st.session_state.messages = []
        st.rerun()

# ── Input form (sticky) ───────────────────────────────────────────────────────
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "msg",
            placeholder="Ex : Quels sont les employés de l'Atlas Machinery ?",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Envoyer")

if submitted and user_input.strip():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner(""):
        answer = chain.invoke(user_input)
    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.rerun()