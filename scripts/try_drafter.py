import pandas as pd
from app.classifier import classify
from app.drafter import draft_reply

TEXT = "consumer_complaint_narrative"
test = pd.read_parquet("data/processed/test.parquet").sample(2, random_state=7)

for _, row in test.iterrows():
    c = classify(row[TEXT])
    d = draft_reply(row[TEXT], c.get("product") or "Unknown", c.get("summary") or "")
    print("=" * 80)
    print("PRODUCT:", c.get("product"), "| SUMMARY:", c.get("summary"))
    print("\nREPLY:\n", d.get("reply"))
    print("\nCITATIONS USED:", d.get("citations_used"),
          "| INVALID:", d.get("invalid_citations"))
    for i, s in enumerate(d["sources"], start=1):
        print(f"  [{i}] {s['doc']} p.{s['page']} (dist {s['distance']}): {s['text'][:120]}...")
    print()