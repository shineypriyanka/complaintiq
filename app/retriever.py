from pathlib import Path
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

ROOT = Path(__file__).resolve().parent.parent   # project root, wherever it runs
DB_DIR = str(ROOT / "data" / "chroma")
COMPLAINTS = "complaints_train"
POLICIES = "rbi_policies"
_ef = None

def get_model():
    global _ef
    if _ef is None:
        # all-MiniLM-L6-v2 served through ONNX Runtime: same model, no PyTorch
        _ef = DefaultEmbeddingFunction()
    return _ef

def embed(texts: list[str]) -> list[list[float]]:
    vecs = np.array(get_model()(texts), dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)   # unit length
    return vecs.tolist()

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