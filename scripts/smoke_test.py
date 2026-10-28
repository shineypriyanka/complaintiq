import time
import numpy as np
from sentence_transformers import SentenceTransformer
from app.llm import chat
from app.config import EMBED_MODEL

# 1. LLM call + latency
t0 = time.perf_counter()
reply = chat("Classify this bank complaint in 3 words: "
             "'I was charged twice for the same ATM withdrawal.'")
print(f"LLM reply: {reply}")
print(f"LLM latency: {time.perf_counter() - t0:.2f}s")

# 2. Embeddings + similarity
model = SentenceTransformer(EMBED_MODEL)
sents = ["My card was charged twice",
         "Duplicate debit on my account",
         "I want to change my home address"]
emb = model.encode(sents, normalize_embeddings=True)
print("Embedding shape:", emb.shape)
print("Sim(0,1) related:  ", round(float(np.dot(emb[0], emb[1])), 3))
print("Sim(0,2) unrelated:", round(float(np.dot(emb[0], emb[2])), 3))