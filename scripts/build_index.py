import pandas as pd
from app.retriever import get_collection, embed

TEXT = "consumer_complaint_narrative"
BATCH = 1000

train = pd.read_parquet("data/processed/train.parquet")
col = get_collection()

ids = train["complaint_id"].astype(str).tolist()
docs = train[TEXT].tolist()
metas = [{"product": p} for p in train["product"]]
embs = embed(docs)

for i in range(0, len(ids), BATCH):
    col.upsert(ids=ids[i:i+BATCH], documents=docs[i:i+BATCH],
               metadatas=metas[i:i+BATCH], embeddings=embs[i:i+BATCH])

print(f"Index size: {col.count():,} complaints")