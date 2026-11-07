import re
import pandas as pd
from pypdf import PdfReader
from app.retriever import ROOT, COMPLAINTS, POLICIES, get_client, embed

TEXT = "consumer_complaint_narrative"
CHUNK_WORDS, OVERLAP, MIN_WORDS = 180, 40, 30
DOC_NAMES = {
    "rbios_2026_faqs.pdf": "RB-IOS 2026 FAQs",
    "rbios_2026_scheme.pdf": "RB-IOS 2026 Scheme",
}

def _fresh_collection(name: str):
    client = get_client()
    try:
        client.delete_collection(name)   # full rebuild: no stale vectors
    except Exception:
        pass
    return client.get_or_create_collection(name)

def build_complaint_index(batch: int = 1000) -> int:
    train = pd.read_parquet(ROOT / "data" / "processed" / "train.parquet")
    col = _fresh_collection(COMPLAINTS)
    ids = train["complaint_id"].astype(str).tolist()
    docs = train[TEXT].tolist()
    metas = [{"product": p} for p in train["product"]]
    embs = embed(docs)
    for i in range(0, len(ids), batch):
        col.add(ids=ids[i:i+batch], documents=docs[i:i+batch],
                metadatas=metas[i:i+batch], embeddings=embs[i:i+batch])
    return col.count()

def _chunks(words: list[str]):
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, max(len(words) - OVERLAP, 1), step):
        yield " ".join(words[start:start + CHUNK_WORDS])

def build_policy_index() -> int:
    col = _fresh_collection(POLICIES)
    ids, docs, metas = [], [], []
    for pdf in sorted((ROOT / "data" / "policies").glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            words = re.sub(r"\s+", " ", page.extract_text() or "").split()
            for i, chunk in enumerate(_chunks(words)):
                if len(chunk.split()) < MIN_WORDS:
                    continue
                ids.append(f"{pdf.stem}-p{page_no}-c{i}")
                docs.append(chunk)
                metas.append({"doc": DOC_NAMES.get(pdf.name, pdf.stem), "page": page_no})
    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
    return col.count()

def ensure_indexes() -> dict:
    """Build any missing index. Fast no-op when both already exist."""
    client = get_client()
    sizes = {name: client.get_or_create_collection(name).count()
             for name in (COMPLAINTS, POLICIES)}
    if sizes[COMPLAINTS] == 0:
        sizes[COMPLAINTS] = build_complaint_index()
    if sizes[POLICIES] == 0:
        sizes[POLICIES] = build_policy_index()
    return {"complaints": sizes[COMPLAINTS], "policies": sizes[POLICIES]}