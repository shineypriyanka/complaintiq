import chromadb
from sentence_transformers import SentenceTransformer
from app.config import EMBED_MODEL

DB_DIR = "data/chroma"
COMPLAINTS = "complaints_train"
POLICIES = "rbi_policies"
_model = None

def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True, batch_size=64,
                              show_progress_bar=len(texts) > 100).tolist()

def get_client():
    return chromadb.PersistentClient(path=DB_DIR)

def get_collection(name: str = COMPLAINTS):
    return get_client().get_or_create_collection(name)

def retrieve(query: str, k: int = 5) -> list[dict]:
    res = get_collection(COMPLAINTS).query(query_embeddings=embed([query]), n_results=k)
    return [{"text": d, "product": m["product"], "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                  res["distances"][0])]

def retrieve_policies(query: str, k: int = 4) -> list[dict]:
    res = get_collection(POLICIES).query(query_embeddings=embed([query]), n_results=k)
    return [{"text": d, "doc": m["doc"], "page": m["page"], "distance": round(dist, 3)}
            for d, m, dist in zip(res["documents"][0], res["metadatas"][0],
                                  res["distances"][0])]