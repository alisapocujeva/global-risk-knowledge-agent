import os
from pathlib import Path

import pypdf
from chromadb import PersistentClient
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

DOCS_DIR = Path("/app/internal_docs")
INDEX_DIR = Path("/app/index")
COLLECTION = "risk_docs"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def iter_pdf_chunks():
    for fname in sorted(DOCS_DIR.iterdir()):
        if not fname.is_file() or fname.suffix.lower() != ".pdf":
            continue
        pdf = pypdf.PdfReader(str(fname))
        for page_number, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip().replace("\n", " ")
            if text:
                yield fname.name, page_number, text


def main():
    if not DOCS_DIR.exists():
        raise FileNotFoundError(f"Docs directory not found: {DOCS_DIR}")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading MiniLM-L6 model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    docs = list(iter_pdf_chunks())
    if not docs:
        raise RuntimeError("No PDF text chunks found to index.")
    print(f"Loaded {len(docs)} PDF text chunks.")

    texts = [chunk[2] for chunk in docs]
    embeddings = model.encode(texts, show_progress_bar=False)
    embeddings = embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings
    metadatas = [{"source": fname, "page": page} for fname, page, _ in docs]
    ids = [f"id_{i}" for i in range(len(texts))]

    print(f"Writing collection '{COLLECTION}' to {INDEX_DIR} (Chroma v2).")
    client = PersistentClient(
        path=str(INDEX_DIR),
        settings=Settings(allow_reset=True, anonymized_telemetry=False),
    )
    existing = {c.name for c in client.list_collections()}
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
    collection = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)
    print("Chroma index build complete.")


if __name__ == "__main__":
    main()
