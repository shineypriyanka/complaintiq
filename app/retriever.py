import chromadb
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL

DB_DIR = "data/chroma"
COLLECTION = "complaints_train"
_model = None

def get_model():
    global _model
    if _model is None:                 # load once, reuse (loading takes seconds)
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64,
                              show_progress_bar=len(texts) > 100).tolist()

def get_collection():
    client = chromadb.PersistentClient(path=DB_DIR)
    return client.get_or_create_collection(COLLECTION)

def retrieve(query: str, k: int = 5) -> list[dict]:
    res = get_collection().query(query_embeddings=embed([query]), n_results=k)
    return [{"text": d, "product": m["product"], "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                  res["distances"][0])]