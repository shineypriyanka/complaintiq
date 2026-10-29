import re
from pathlib import Path
from pypdf import PdfReader
from app.retriever import get_client, embed, POLICIES

CHUNK_WORDS, OVERLAP, MIN_WORDS = 180, 40, 30
DOC_NAMES = {
    "rbios_2026_faqs.pdf": "RB-IOS 2026 FAQs",
    "rbios_2026_scheme.pdf": "RB-IOS 2026 Scheme",
}

def chunks(words: list[str]):
    step = CHUNK_WORDS - OVERLAP
    for start in range(0, max(len(words) - OVERLAP, 1), step):
        yield " ".join(words[start:start + CHUNK_WORDS])

def main():
    client = get_client()
    try:
        client.delete_collection(POLICIES)   # full rebuild: no stale chunks
    except Exception:
        pass
    col = client.get_or_create_collection(POLICIES)

    ids, docs, metas = [], [], []
    for pdf in sorted(Path("data/policies").glob("*.pdf")):
        for page_no, page in enumerate(PdfReader(pdf).pages, start=1):
            words = re.sub(r"\s+", " ", page.extract_text() or "").split()
            for i, chunk in enumerate(chunks(words)):
                if len(chunk.split()) < MIN_WORDS:
                    continue   # skip headers, footers, near-empty pages
                ids.append(f"{pdf.stem}-p{page_no}-c{i}")
                docs.append(chunk)
                metas.append({"doc": DOC_NAMES.get(pdf.name, pdf.stem), "page": page_no})

    col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embed(docs))
    print(f"Indexed {col.count()} policy chunks")
    print("Sample:", metas[0], docs[0][:200])

if __name__ == "__main__":
    main()