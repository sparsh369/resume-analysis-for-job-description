# ResumeRAG — Streamlit App

A RAG-powered resume analyzer and job matcher. Upload a PDF resume, add job descriptions, and get ranked match scores with skill gap analysis.

## How it works

1. **PDF Extraction** — PyMuPDF pulls raw text from your resume
2. **Chunking** — Sliding window (150 tokens, 30 overlap) splits text into semantic segments
3. **TF-IDF Embeddings** — Each chunk gets an 80-dim vector; job descriptions get the same treatment
4. **Retrieval** — Cosine similarity finds the top-5 resume chunks most relevant to each job
5. **LLM Matching** — Claude receives only the retrieved chunks (not the full resume) and generates grounded match analysis

## Local setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Cloud

1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io → New app
3. Select your repo, branch, and set **Main file path** to `app.py`
4. Click Deploy — no secrets needed (API key entered at runtime in the UI)

## File structure

```
resume_rag/
├── app.py               # Main Streamlit app
├── requirements.txt     # Python dependencies
├── .streamlit/
│   └── config.toml      # Theme configuration
└── README.md
```

## Chunking defaults (tuned for resumes)

| Parameter  | Value | Reason |
|------------|-------|--------|
| Chunk size | 150 tokens | Captures a full job role or skills section without mixing contexts |
| Overlap    | 30 tokens  | 20% overlap prevents splitting role titles from their descriptions |
| Top-K      | 5 chunks   | Enough context without diluting relevance |
| Vocab size | 80 terms   | Covers all meaningful resume keywords, ignores stopwords |
