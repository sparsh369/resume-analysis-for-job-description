import streamlit as st
from openai import OpenAI
import fitz
import numpy as np
import math
import re
from collections import Counter

# ── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeRAG · Job Matcher",
    page_icon="⚡",
    layout="wide",
)

# ── API KEY ─────────────────────────────────────────────
OPENAI_API_KEY = ""
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass

# ── RAG CONFIG ─────────────────────────────────────────────
CHUNK_SIZE = 150
OVERLAP = 30
TOP_K = 5
VOCAB_SIZE = 80

# ── UI FIX (IMPORTANT) ─────────────────────────────────────
st.markdown("""
<style>

/* GLOBAL FIX (removes faded UI issue) */
html, body, .stApp {
    background: #F4F6FB !important;
    color: #1A1D27 !important;
}

/* Force all text visible */
p, span, label, div, h1, h2, h3 {
    color: #1A1D27 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #E5E7EB;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #F8F9FB !important;
    border: 1px solid #E2E4EC !important;
    border-radius: 10px !important;
}

/* Inputs */
input, textarea {
    background: #FFFFFF !important;
    color: #111827 !important;
}

/* Buttons */
.stButton > button {
    background: #FFFFFF !important;
    color: #1A1D27 !important;
    border: 1px solid #D1D5DB !important;
    border-radius: 8px;
}
.stButton > button:hover {
    background: #EEF2FF !important;
    color: #4F46E5 !important;
}

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(135deg, #4F46E5, #7C3AED) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 20px rgba(79,70,229,0.3);
}

/* Cards */
.match-card {
    background: #FFFFFF !important;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
    border: 1px solid #E5E7EB;
    box-shadow: 0 10px 25px rgba(0,0,0,0.06);
    transition: 0.2s;
}
.match-card:hover {
    transform: translateY(-4px);
}

/* Header */
.main-title {
    font-size: 28px;
    font-weight: 700;
    margin-bottom: 10px;
}

.sub-text {
    color: #6B7280;
    font-size: 14px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ── FUNCTIONS ──────────────────────────────────────────────
def extract_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    return " ".join(p.get_text() for p in doc)

def chunk_text(text):
    words = text.split()
    chunks = []
    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
    return chunks

def build_vocab(docs):
    cnt = Counter(w for d in docs for w in re.findall(r"[a-z]{3,}", d.lower()))
    return [w for w, _ in cnt.most_common(VOCAB_SIZE)]

def tfidf_vec(text, vocab, docs):
    tokens = re.findall(r"[a-z]{3,}", text.lower())
    tc = Counter(tokens)
    n = len(tokens) or 1
    return np.array([
        (tc[w]/n) * (math.log((len(docs)+1)/(1+sum(w in d for d in docs)))+1)
        for w in vocab
    ])

def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a)*np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0

def call_gpt(prompt):
    client = OpenAI(api_key=OPENAI_API_KEY)
    res = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500
    )
    return res.choices[0].message.content

# ── SIDEBAR ────────────────────────────────────────────────
st.sidebar.markdown("## ⚡ ResumeRAG")

pdf = st.sidebar.file_uploader("Upload Resume (PDF)", type="pdf")

job_title = st.sidebar.text_input("Job Title")
job_desc = st.sidebar.text_area("Job Description")

jobs = st.session_state.get("jobs", [])

if st.sidebar.button("➕ Add Job"):
    if job_title and job_desc:
        jobs.append({"title": job_title, "desc": job_desc})
        st.session_state.jobs = jobs

if jobs:
    st.sidebar.markdown("### 📌 Jobs Added")
    for j in jobs:
        st.sidebar.write("•", j["title"])

# ── RUN BUTTON ─────────────────────────────────────────────
missing = []
if not OPENAI_API_KEY:
    missing.append("API key")
if not pdf:
    missing.append("Resume PDF")
if not jobs:
    missing.append("Job description")

run = st.sidebar.button("🚀 Analyze", disabled=len(missing) > 0)

if missing:
    st.sidebar.warning("Need: " + ", ".join(missing))

# ── MAIN ───────────────────────────────────────────────────
st.markdown('<div class="main-title">Job Match Analysis</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI-powered resume matching</div>', unsafe_allow_html=True)

if run:
    with st.spinner("Analyzing..."):

        text = extract_pdf(pdf)
        chunks = chunk_text(text)
        vocab = build_vocab(chunks)
        chunk_vecs = [tfidf_vec(c, vocab, chunks) for c in chunks]

        results = []

        for job in jobs:
            jv = tfidf_vec(job["desc"], vocab, chunks)
            sims = [cosine(jv, cv) for cv in chunk_vecs]
            top_chunks = sorted(zip(chunks, sims), key=lambda x: -x[1])[:TOP_K]

            context = "\n".join(c for c, _ in top_chunks)

            prompt = f"""
Resume:
{context}

Job:
{job['desc']}

Give:
- Score out of 100
- Summary
- Matching skills
- Missing skills
"""

            out = call_gpt(prompt)

            results.append({
                "title": job["title"],
                "response": out
            })

    # ── DISPLAY ────────────────────────────────────────────
    for r in results:
        st.markdown(f"### 💼 {r['title']}")
        st.markdown(f"<div class='match-card'>{r['response']}</div>", unsafe_allow_html=True)
