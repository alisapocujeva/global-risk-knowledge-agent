Folder structure:

global_risk_agent/
│
├── Dockerfile
├── requirements.txt
├── build_index.py
│
├── internal_docs/   → 17 PDFs
├── index/           → auto-built by Chroma
│
└── app/
    ├── app.py
    └── agent.py

Key requirements:
- Perplexity API via official HTTP format
- Perplexity model: llama-3.1-sonar-large-128k-online
- Internal search uses Chroma + MiniLM-L6
- External search: Perplexity API
- UI: Streamlit

Everything else must be removed.
