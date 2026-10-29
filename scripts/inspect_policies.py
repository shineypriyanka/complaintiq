from pathlib import Path
from pypdf import PdfReader

for pdf in sorted(Path("data/policies").glob("*.pdf")):
    reader = PdfReader(pdf)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    print(f"=== {pdf.name}: {len(reader.pages)} pages, {len(text.split()):,} words ===")
    print(text[:800])
    print("...\n")