import pandas as pd
from app.classifier import classify

TEXT, LABEL = "consumer_complaint_narrative", "product"
test = pd.read_parquet("data/processed/test.parquet").sample(5, random_state=1)

for _, row in test.iterrows():
    r = classify(row[TEXT])
    mark = "OK " if r.get("product") == row[LABEL] else "XX "
    print(f"{mark} true={row[LABEL]}\n    pred={r.get('raw_product')}")
    print(f"    severity={r.get('severity')} | {r.get('latency_s')}s | "
          f"tokens in/out={r.get('prompt_tokens')}/{r.get('completion_tokens')}")
    print(f"    summary={r.get('summary')}\n")