import streamlit as st
from openai import OpenAI
import fitz  # PyMuPDF
import numpy as np
import math
import re
import json
from collections import Counter

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeRAG · Job Matcher",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── API key: Streamlit secrets first, fallback to sidebar input ───────────────
# To add your key as a secret:
#   LOCAL:  create .streamlit/secrets.toml and add:
#           [openai]
#           api_key = "sk-..."
#   CLOUD:  Streamlit Cloud → App settings → Secrets → paste the same toml block
OPENAI_KEY = ""
try:
    OPENAI_KEY = st.secrets["openai"]["api_key"]
except Exception:
    pass

# ── RAG constants ─────────────────────────────────────────────────────────────
CHUNK_SIZE = 150   # tokens per chunk — captures one role/skills section cleanly
OVERLAP    = 30    # 20% overlap prevents splitting role titles from descriptions
TOP_K      = 5     # top chunks retrieved per job query
VOCAB_SIZE = 80    # TF-IDF vocabulary size

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

.main, .stApp { background: #F8F9FB !important; }
.block-container { padding: 2rem 2.5rem 3rem !important; max-width: 1160px !important; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E8EAF0 !important;
}
section[data-testid="stSidebar"] .block-container { padding: 0 !important; }

/* All sidebar text must be dark — override Streamlit dark mode leakage */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] .stMarkdown {
    color: #1A1D27 !important;
}
section[data-testid="stSidebar"] .stTextInput label,
section[data-testid="stSidebar"] .stTextArea label,
section[data-testid="stSidebar"] .stFileUploader label {
    color: #1A1D27 !important;
}
section[data-testid="stSidebar"] .stTextInput input,
section[data-testid="stSidebar"] .stTextArea textarea {
    background: #F8F9FB !important;
    border: 1px solid #E2E4EC !important;
    color: #1A1D27 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}
section[data-testid="stSidebar"] .stTextInput input::placeholder,
section[data-testid="stSidebar"] .stTextArea textarea::placeholder {
    color: #A0A4B8 !important;
}
section[data-testid="stSidebar"] .stTextInput input:focus,
section[data-testid="stSidebar"] .stTextArea textarea:focus {
    border-color: #4F6EF7 !important;
    box-shadow: 0 0 0 3px rgba(79,110,247,0.1) !important;
    outline: none !important;
}

/* ── All buttons: white bg, dark text, clear border ── */
.stButton > button {
    background: #FFFFFF !important;
    color: #1A1D27 !important;
    border: 1.5px solid #D0D3E0 !important;
    border-radius: 8px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    transition: all 0.18s !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
}
.stButton > button:hover:not(:disabled) {
    background: #EEF1FF !important;
    border-color: #4F6EF7 !important;
    color: #4F6EF7 !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(79,110,247,0.15) !important;
}
.stButton > button:disabled {
    background: #F4F5F8 !important;
    border-color: #E2E4EC !important;
    color: #B0B4C8 !important;
    cursor: not-allowed !important;
    box-shadow: none !important;
}
/* Sidebar small buttons */
section[data-testid="stSidebar"] .stButton > button {
    background: #F8F9FB !important;
    color: #1A1D27 !important;
    border: 1.5px solid #E2E4EC !important;
    font-size: 0.8rem !important;
    box-shadow: none !important;
}
section[data-testid="stSidebar"] .stButton > button:hover:not(:disabled) {
    background: #EEF1FF !important;
    border-color: #4F6EF7 !important;
    color: #4F6EF7 !important;
    transform: none !important;
    box-shadow: none !important;
}
/* Primary run button — always blue fill */
button[kind="primary"],
.stButton > button[kind="primary"],
section[data-testid="stSidebar"] button[kind="primary"],
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background: #4F6EF7 !important;
    color: #FFFFFF !important;
    border: none !important;
    font-size: 0.92rem !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(79,110,247,0.3) !important;
}
button[kind="primary"]:hover:not(:disabled),
section[data-testid="stSidebar"] button[kind="primary"]:hover:not(:disabled) {
    background: #3B5CE8 !important;
    color: #FFFFFF !important;
    border: none !important;
    transform: none !important;
    box-shadow: 0 6px 18px rgba(79,110,247,0.4) !important;
}
button[kind="primary"]:disabled {
    background: #B8C4FB !important;
    color: #FFFFFF !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Sidebar components ── */
.sb-logo {
    padding: 24px 22px 18px;
    border-bottom: 1px solid #F0F1F6;
}
.sb-logo-name { font-size: 1.25rem; font-weight: 700; color: #1A1D27; }
.sb-logo-name span { color: #4F6EF7; }
.sb-logo-sub  { font-size: 0.7rem; color: #8B90A7; margin-top: 2px; }

.sb-section { padding: 16px 22px 0; }
.sb-step {
    font-size: 0.62rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #8B90A7;
    margin-bottom: 10px;
    display: flex; align-items: center; gap: 7px;
}
.sb-step-num {
    width: 17px; height: 17px;
    background: #1A1D27; color: white;
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 9px; font-weight: 700; flex-shrink: 0;
}
.sb-hr { height: 1px; background: #F0F1F6; margin: 16px 22px; }

.sb-file-ok {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem; color: #1A7A4A;
    background: #EDFAF4; border: 1px solid #B6EDD3;
    border-radius: 8px; padding: 8px 12px;
    margin-top: 6px;
}
.sb-file-name { flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.sb-file-size { font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; color: #8B90A7; }

.sb-key-ok {
    display: flex; align-items: center; gap: 8px;
    font-size: 0.78rem; color: #1A7A4A;
    background: #EDFAF4; border: 1px solid #B6EDD3;
    border-radius: 8px; padding: 8px 12px;
}

.q-item {
    display: flex; align-items: center; gap: 9px;
    padding: 8px 12px;
    background: #F8F9FB; border: 1px solid #E8EAF0;
    border-radius: 8px; margin-bottom: 5px;
}
.q-dot  { width: 6px; height: 6px; border-radius: 50%; background: #4F6EF7; flex-shrink: 0; }
.q-name { font-size: 0.8rem; font-weight: 500; color: #1A1D27; flex: 1;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.q-idx  { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; color: #A0A4B8; }

/* ── Top bar ── */
.top-bar {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid #E8EAF0;
}
.top-title { font-size: 1.45rem; font-weight: 700; color: #1A1D27; }
.top-sub   { font-size: 0.8rem; color: #8B90A7; margin-top: 2px; }
.top-badge {
    font-size: 0.7rem; font-weight: 500; color: #8B90A7;
    background: white; border: 1px solid #E8EAF0;
    padding: 5px 13px; border-radius: 20px;
    display: inline-flex; align-items: center; gap: 7px;
}
.live-dot { width: 7px; height: 7px; border-radius: 50%; background: #22C55E; display: inline-block; }

/* ── Empty state ── */
.empty-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 80px 40px; text-align: center; min-height: 55vh;
}
.empty-icon  { font-size: 2.8rem; opacity: 0.2; margin-bottom: 16px; }
.empty-h     { font-size: 1.05rem; font-weight: 600; color: #1A1D27; margin-bottom: 6px; }
.empty-p     { font-size: 0.83rem; color: #8B90A7; line-height: 1.65; }

/* ── Candidate profile card ── */
.profile-card {
    background: white; border: 1px solid #E8EAF0; border-radius: 14px;
    padding: 24px 28px; margin-bottom: 22px;
    display: flex; align-items: center; gap: 22px; flex-wrap: wrap;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.profile-av {
    width: 52px; height: 52px;
    background: linear-gradient(135deg, #4F6EF7 0%, #7C3AED 100%);
    border-radius: 14px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; font-weight: 700; color: white; flex-shrink: 0;
    letter-spacing: -0.5px;
}
.profile-name { font-size: 1.25rem; font-weight: 700; color: #1A1D27; margin-bottom: 3px; }
.profile-role { font-size: 0.7rem; font-weight: 600; letter-spacing: 1px;
                text-transform: uppercase; color: #4F6EF7; }
.profile-skills { display: flex; flex-wrap: wrap; gap: 6px; flex: 1; min-width: 200px; }
.skill-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
    background: #EEF1FF; color: #3451C7; border: 1px solid #D4DCFF;
}

/* ── Stats row ── */
.stats-row { display: flex; gap: 14px; margin-bottom: 24px; flex-wrap: wrap; }
.stat-box {
    flex: 1; min-width: 110px;
    background: white; border: 1px solid #E8EAF0; border-radius: 12px;
    padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.stat-v { font-size: 1.75rem; font-weight: 700; color: #1A1D27; line-height: 1; margin-bottom: 4px;
          font-family: 'JetBrains Mono', monospace; }
.stat-l { font-size: 0.68rem; font-weight: 500; color: #8B90A7;
          text-transform: uppercase; letter-spacing: 0.5px; }

/* ── Section label ── */
.sec-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 1.5px;
    text-transform: uppercase; color: #8B90A7;
    margin-bottom: 16px;
    display: flex; align-items: center; gap: 10px;
}
.sec-label::after { content: ''; flex: 1; height: 1px; background: #E8EAF0; }

/* ── Match card ── */
.match-card {
    background: white; border: 1px solid #E8EAF0; border-radius: 14px;
    overflow: hidden; margin-bottom: 14px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    transition: box-shadow 0.2s;
}
.match-card:hover { box-shadow: 0 6px 20px rgba(0,0,0,0.08); }
.match-card.best  { border-color: #4F6EF7; border-width: 1.5px; }

.match-header { display: flex; align-items: stretch; }
.match-rank-col {
    width: 62px; background: #F8F9FB; border-right: 1px solid #F0F1F6;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 14px 8px; flex-shrink: 0;
}
.rank-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem; font-weight: 700; color: #D0D3E0; line-height: 1;
}
.rank-num.best { color: #4F6EF7; }

.match-info-col { flex: 1; padding: 16px 20px; }
.m-job-title { font-size: 1rem; font-weight: 700; color: #1A1D27; margin-bottom: 6px; }
.m-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.4px;
    text-transform: uppercase; padding: 2px 9px; border-radius: 20px;
    display: inline-block;
}
.best-badge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.65rem; font-weight: 600; letter-spacing: 0.6px;
    text-transform: uppercase; color: #4F6EF7;
    background: #EEF1FF; border: 1px solid #C5CEFF;
    padding: 2px 9px; border-radius: 20px; margin-right: 7px;
}

.match-score-col {
    min-width: 96px; background: #F8F9FB; border-left: 1px solid #F0F1F6;
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 14px 18px;
}
.sc-num { font-family: 'JetBrains Mono', monospace; font-size: 2.2rem; font-weight: 700; line-height: 1; }
.sc-bar { width: 52px; height: 3px; background: #E8EAF0; border-radius: 2px; margin: 5px 0; }
.sc-fill { height: 100%; border-radius: 2px; }
.sc-cap { font-size: 0.58rem; font-weight: 600; letter-spacing: 1px;
          text-transform: uppercase; color: #8B90A7; }

.match-body { padding: 16px 20px; border-top: 1px solid #F0F1F6; }
.m-summary { font-size: 0.86rem; color: #3D4152; line-height: 1.75; margin-bottom: 14px; }
.tags-lbl {
    font-size: 0.6rem; font-weight: 600; letter-spacing: 1.2px;
    text-transform: uppercase; color: #A0A4B8; margin-bottom: 6px;
}
.tags-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; font-weight: 500;
    padding: 3px 9px; border-radius: 4px; border: 1px solid;
}
.tag-hit { color: #1A7A4A; background: #EDFAF4; border-color: #B6EDD3; }
.tag-gap { color: #B94040; background: #FDF0F0; border-color: #F5C2C2; }
.reco {
    margin-top: 12px; padding: 11px 15px;
    background: #F3F5FF; border-left: 3px solid #4F6EF7;
    border-radius: 0 8px 8px 0;
    font-size: 0.82rem; color: #2D3880; line-height: 1.6;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
.stDeployButton { display: none !important; }
div[data-testid="stDecoration"] { display: none !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #E2E4EC; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"jobs": [], "results": None, "ran": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── RAG pipeline functions ────────────────────────────────────────────────────
def extract_pdf(file) -> str:
    doc  = fitz.open(stream=file.read(), filetype="pdf")
    text = " ".join(p.get_text() for p in doc)
    return re.sub(r"\s+", " ", text).strip()

def chunk_text(text: str) -> list[dict]:
    words  = text.split()
    chunks, start, idx = [], 0, 0
    while start < len(words):
        end = min(start + CHUNK_SIZE, len(words))
        chunks.append({"id": f"chunk_{idx:03d}", "text": " ".join(words[start:end]),
                        "tokens": end - start})
        idx += 1
        if end >= len(words): break
        start = end - OVERLAP
    return chunks

def build_vocab(docs: list[str]) -> list[str]:
    stop = {"the","a","an","and","or","in","of","to","for","is","are","was","were",
            "be","been","with","at","by","from","on","as","its","this","that","it","we","you","i"}
    cnt  = Counter(w for d in docs for w in re.findall(r"[a-z]{3,}", d.lower()) if w not in stop)
    return [w for w, _ in cnt.most_common(VOCAB_SIZE)]

def tfidf_vec(text: str, vocab: list[str], docs: list[str]) -> np.ndarray:
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    tc = Counter(tokens); n = len(tokens) or 1
    return np.array([
        (tc[w]/n) * (math.log((len(docs)+1)/(sum(1 for d in docs if w in re.findall(r"[a-z]{3,}", d.lower()))+1))+1)
        for w in vocab
    ], dtype=float)

def cosine(a, b) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a,b)/(na*nb)) if na and nb else 0.0

def retrieve(qv, cvecs, k):
    return sorted(enumerate(cvecs), key=lambda x: -cosine(qv, x[1]))[:k]

def call_gpt(prompt: str, key: str) -> str:
    client = OpenAI(api_key=key)
    r = client.chat.completions.create(
        model="gpt-4o", max_tokens=3000,
        messages=[{"role": "user", "content": prompt}]
    )
    return r.choices[0].message.content

def score_color(s: int) -> str:
    if s >= 75: return "#1A7A4A"
    if s >= 60: return "#4F6EF7"
    if s >= 45: return "#D97706"
    return "#B94040"

def label_style(s: int) -> tuple[str, str, str]:
    if s >= 75: return "#1A7A4A", "#EDFAF4", "#B6EDD3"
    if s >= 60: return "#3451C7", "#EEF1FF", "#C5CEFF"
    if s >= 45: return "#92400E", "#FFF8ED", "#FCD99F"
    return "#B94040", "#FDF0F0", "#F5C2C2"

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:

    st.markdown("""
    <div class="sb-logo">
        <div class="sb-logo-name">Resume<span>RAG</span></div>
        <div class="sb-logo-sub">AI-powered job match intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key — always from secrets, never shown in UI ─────────────────────
    api_key = OPENAI_KEY  # loaded from st.secrets["openai"]["api_key"]

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

    # ── PDF Upload ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section"><div class="sb-step"><span class="sb-step-num">1</span>Resume PDF</div>', unsafe_allow_html=True)
    pdf_file = st.file_uploader("PDF", type="pdf", label_visibility="collapsed")
    st.markdown('</div>', unsafe_allow_html=True)

    if pdf_file:
        st.markdown(f"""
        <div style="padding: 6px 22px 0;">
            <div class="sb-file-ok">
                <span>📄</span>
                <span class="sb-file-name" style="color:#1A7A4A !important;">{pdf_file.name[:26]}{'…' if len(pdf_file.name)>26 else ''}</span>
                <span class="sb-file-size">{pdf_file.size//1024}kb</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)

    # ── Job Descriptions ───────────────────────────────────────────────────────
    st.markdown('<div class="sb-section"><div class="sb-step"><span class="sb-step-num">2</span>Job Descriptions</div>', unsafe_allow_html=True)
    j_title = st.text_input("Job title", placeholder="e.g. Senior React Developer",
                             label_visibility="collapsed")
    j_desc  = st.text_area("Job description", placeholder="Paste job description here…",
                            height=108, label_visibility="collapsed")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("+ Add job", use_container_width=True):
            if j_title.strip() and j_desc.strip():
                st.session_state.jobs.append({
                    "id": int(__import__("time").time() * 1000),
                    "title": j_title.strip(), "desc": j_desc.strip()
                })
                st.rerun()
    with col2:
        if st.button("Load samples", use_container_width=True):
            st.session_state.jobs = [
                {"id": 1, "title": "Senior Frontend Engineer",
                 "desc": "5+ yrs React & TypeScript. State management Redux/Zustand. GraphQL, REST APIs. Core Web Vitals. Next.js, AWS, CI/CD pipelines. Mentor junior devs, lead architecture decisions."},
                {"id": 2, "title": "Full Stack Developer",
                 "desc": "Node.js, React, PostgreSQL, AWS, Docker. 3+ yrs full-stack. Microservices architecture. CI/CD required. TypeScript. Bonus: Kafka, Kubernetes, startup experience."},
                {"id": 3, "title": "ML Engineer",
                 "desc": "Python, PyTorch or TensorFlow. NLP, LLM fine-tuning. Vector databases Pinecone/Weaviate. MLOps: MLflow, DVC. AWS SageMaker. RAG pipelines. Production model deployment."},
            ]
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Job queue ──────────────────────────────────────────────────────────────
    if st.session_state.jobs:
        st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="sb-section"><div class="sb-step"><span class="sb-step-num">{len(st.session_state.jobs)}</span>Queued</div>', unsafe_allow_html=True)
        for idx, j in enumerate(st.session_state.jobs):
            c1, c2 = st.columns([6, 1])
            c1.markdown(f"""
            <div class="q-item">
                <div class="q-dot"></div>
                <div class="q-name">{j['title']}</div>
                <div class="q-idx">#{idx+1}</div>
            </div>""", unsafe_allow_html=True)
            if c2.button("✕", key=f"del_{j['id']}"):
                st.session_state.jobs = [x for x in st.session_state.jobs if x["id"] != j["id"]]
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Run button ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-hr"></div>', unsafe_allow_html=True)
    can_run = bool(pdf_file and st.session_state.jobs and api_key)

    with st.container():
        st.markdown('<div style="padding: 0 22px 24px;">', unsafe_allow_html=True)
        run_clicked = st.button(
            "⚡  Analyze & Match",
            use_container_width=True,
            type="primary",
            disabled=not can_run
        )
        if not can_run:
            missing = [x for x, y in [("API key", api_key), ("PDF", pdf_file),
                                        ("job description", st.session_state.jobs)] if not y]
            st.markdown(f'<div style="font-size:0.7rem;color:#A0A4B8;text-align:center;margin-top:6px;">'
                        f'Need: {" · ".join(missing)}</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ── MAIN CONTENT ──────────────────────────────────────────────────────────────

st.markdown("""
<div class="top-bar">
    <div>
        <div class="top-title">Job Match Analysis</div>
        <div class="top-sub">RAG-powered resume intelligence</div>
    </div>
    <div class="top-badge">
        <span class="live-dot"></span>
        GPT-4o · TF-IDF RAG pipeline
    </div>
</div>
""", unsafe_allow_html=True)

# ── Run pipeline (all phases silently in background) ─────────────────────────
if run_clicked:
    st.session_state.results = None
    st.session_state.ran     = True

    with st.spinner("Running RAG pipeline — extracting, embedding, retrieving, matching…"):
        try:
            # Phase 1–3: silent
            raw_text   = extract_pdf(pdf_file)
            chunks     = chunk_text(raw_text)
            docs       = [c["text"] for c in chunks]
            vocab      = build_vocab(docs)
            chunk_vecs = [tfidf_vec(c["text"], vocab, docs) for c in chunks]

            # Phase 4: retrieve + GPT-4o
            blocks = ""
            for i, job in enumerate(st.session_state.jobs):
                jv   = tfidf_vec(job["desc"], vocab, docs)
                top  = retrieve(jv, chunk_vecs, min(TOP_K, len(chunks)))
                ctx  = "\n---\n".join(chunks[idx]["text"] for idx, _ in top)
                sims = ", ".join(f"{cosine(jv, chunk_vecs[idx]):.3f}" for idx, _ in top)
                blocks += f"""
<job index="{i+1}">
<title>{job['title']}</title>
<description>{job['desc']}</description>
<retrieved_resume_chunks similarity_scores="{sims}">
{ctx}
</retrieved_resume_chunks>
</job>
"""

            prompt = f"""You are a precise RAG-powered job matcher. For each job you have the top-{TOP_K} most semantically relevant chunks retrieved from the candidate's resume via TF-IDF cosine similarity. Analyse each job using ONLY the retrieved resume context.

{blocks}

Return ONLY valid JSON — no markdown fences, no preamble:
{{
  "candidate": {{
    "name": "extract from resume or use Candidate",
    "headline": "2-4 word role summary",
    "topSkills": ["list up to 8 skills"]
  }},
  "matches": [
    {{
      "jobIndex": 1,
      "jobTitle": "exact title",
      "score": 82,
      "scoreLabel": "Strong Match",
      "summary": "2-3 sentences grounded strictly in retrieved chunks",
      "matchingSkills": ["skills present in both resume chunks and job"],
      "skillGaps": ["skills the job needs that are absent from resume"],
      "recommendation": "One specific, actionable sentence"
    }}
  ]
}}

Score rubric: 90-100=Perfect Match, 75-89=Strong Match, 60-74=Good Match, 45-59=Partial Match, <45=Weak Match.
Sort matches array by score descending."""

            raw = call_gpt(prompt, api_key)
            hit = re.search(r'\{[\s\S]*\}', raw)
            if hit:
                st.session_state.results = json.loads(hit.group())
            else:
                st.error("Could not parse response. Try again.")
        except Exception as e:
            st.error(f"Pipeline error: {e}")

# ── Empty / waiting state ─────────────────────────────────────────────────────
if not st.session_state.ran:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">⚡</div>
        <div class="empty-h">Ready to analyze your resume</div>
        <div class="empty-p">
            Upload your PDF, add job descriptions,<br>
            then click <strong>Analyze &amp; Match</strong> in the sidebar.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not st.session_state.results:
    if st.session_state.ran and not run_clicked:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">⚠️</div>
            <div class="empty-h">No results returned</div>
            <div class="empty-p">Check your API key and try again.</div>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ── Render results ────────────────────────────────────────────────────────────
parsed  = st.session_state.results
cand    = parsed.get("candidate", {})
matches = sorted(parsed.get("matches", []), key=lambda x: -x.get("score", 0))

# Candidate profile card
initials    = "".join(w[0].upper() for w in cand.get("name","?").split()[:2])
skills_html = "".join(f'<span class="skill-tag">{s}</span>' for s in cand.get("topSkills", []))

st.markdown(f"""
<div class="profile-card">
    <div class="profile-av">{initials}</div>
    <div>
        <div class="profile-name">{cand.get('name', 'Candidate')}</div>
        <div class="profile-role">{cand.get('headline', '')}</div>
    </div>
    <div class="profile-skills">{skills_html}</div>
</div>
""", unsafe_allow_html=True)

# Summary stats
top_sc = matches[0].get("score", 0) if matches else 0
avg_sc = int(sum(m.get("score", 0) for m in matches) / len(matches)) if matches else 0
strong = sum(1 for m in matches if m.get("score", 0) >= 75)

st.markdown(f"""
<div class="stats-row">
    <div class="stat-box">
        <div class="stat-v">{len(matches)}</div>
        <div class="stat-l">Jobs analyzed</div>
    </div>
    <div class="stat-box">
        <div class="stat-v" style="color:#4F6EF7">{top_sc}</div>
        <div class="stat-l">Best score</div>
    </div>
    <div class="stat-box">
        <div class="stat-v">{avg_sc}</div>
        <div class="stat-l">Avg. score</div>
    </div>
    <div class="stat-box">
        <div class="stat-v" style="color:#1A7A4A">{strong}</div>
        <div class="stat-l">Strong matches</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Match cards
st.markdown('<div class="sec-label">Ranked job matches</div>', unsafe_allow_html=True)

for i, m in enumerate(matches):
    sc        = m.get("score", 0)
    col       = score_color(sc)
    tc, bg, bd = label_style(sc)
    is_best   = (i == 0)

    m_tags = "".join(f'<span class="tag tag-hit">✓ {s}</span>' for s in m.get("matchingSkills", []))
    g_tags = "".join(f'<span class="tag tag-gap">△ {s}</span>' for s in m.get("skillGaps", []))
    best_html = '<span class="best-badge">⭐ Best match</span>' if is_best else ""

    st.markdown(f"""
    <div class="match-card {'best' if is_best else ''}">
        <div class="match-header">
            <div class="match-rank-col">
                <div class="rank-num {'best' if is_best else ''}">{str(i+1).zfill(2)}</div>
            </div>
            <div class="match-info-col">
                <div class="m-job-title">{m.get('jobTitle','')}</div>
                <div>
                    {best_html}
                    <span class="m-label" style="color:{tc};background:{bg};border:1px solid {bd}">
                        {m.get('scoreLabel','')}
                    </span>
                </div>
            </div>
            <div class="match-score-col">
                <div class="sc-num" style="color:{col}">{sc}</div>
                <div class="sc-bar"><div class="sc-fill" style="width:{sc}%;background:{col}"></div></div>
                <div class="sc-cap">fit score</div>
            </div>
        </div>
        <div class="match-body">
            <div class="m-summary">{m.get('summary','')}</div>
            {'<div class="tags-lbl">Matching skills</div><div class="tags-row">'+m_tags+'</div>' if m_tags else ''}
            {'<div class="tags-lbl" style="margin-top:10px">Skill gaps</div><div class="tags-row">'+g_tags+'</div>' if g_tags else ''}
            <div class="reco">{m.get('recommendation','')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
