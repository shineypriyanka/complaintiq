import pandas as pd
from app.classifier import classify, classify_rag

TEXT, LABEL = "consumer_complaint_narrative", "product"
test = pd.read_parquet("data/processed/test.parquet").sample(5, random_state=1)

for _, row in test.iterrows():
    zs, rag = classify(row[TEXT]), classify_rag(row[TEXT])
    print(f"true      = {row[LABEL]}")
    print(f"zero-shot = {zs.get('raw_product')}")
    print(f"rag       = {rag.get('raw_product')}")
    print(f"retrieved = {rag.get('retrieved_labels')}")
    print(f"tokens in: zero-shot {zs.get('prompt_tokens')} vs rag {rag.get('prompt_tokens')}\n")