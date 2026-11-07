from contextlib import asynccontextmanager
from typing import Literal
import groq
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.config import LLM_MODEL
from app.classifier import classify, classify_rag, classify_knn
from app.drafter import draft_reply
from app.pii import redact
from app.retriever import get_model, get_collection, COMPLAINTS, POLICIES

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()   # warm start: load the embedding model once, at startup
    app.state.index_sizes = {
        "complaints": get_collection(COMPLAINTS).count(),
        "policies": get_collection(POLICIES).count(),
    }
    yield

app = FastAPI(title="ComplaintIQ API", version="0.1.0", lifespan=lifespan)

# ---------- request / response models (the API contract) ----------
class ComplaintIn(BaseModel):
    text: str = Field(..., min_length=20, max_length=10_000,
                      description="Customer complaint text")
    method: Literal["rag", "zero_shot", "knn"] = "rag"

class ClassifyOut(BaseModel):
    product: str
    severity: str | None = None
    summary: str | None = None
    method_used: str
    degraded: bool = False
    knn_confidence: float | None = None
    redactions: dict[str, int] = {}
    latency_s: float | None = None

class Source(BaseModel):
    doc: str
    page: int
    text: str

class TriageOut(ClassifyOut):
    reply: str | None = None
    citations_used: list[int] = []
    sources: list[Source] = []
    reply_error: str | None = None

# ---------- core logic ----------
METHODS = {"rag": classify_rag, "zero_shot": classify, "knn": classify_knn}

def run_classifier(text: str, method: str) -> tuple[dict, str, bool]:
    """Returns (result, method_actually_used, degraded)."""
    try:
        result = METHODS[method](text)
        if result.get("product"):
            return result, method, False
    except (groq.APIError, RuntimeError):
        pass   # rate limit, outage, bad model name... fall through to kNN
    return classify_knn(text), "knn", method != "knn"

def to_classify_out(r: dict, used: str, degraded: bool, redactions: dict) -> dict:
    return dict(product=r["product"], severity=r.get("severity"),
                summary=r.get("summary"), method_used=used, degraded=degraded,
                knn_confidence=r.get("knn_confidence"), redactions=redactions,
                latency_s=r.get("latency_s"))

# ---------- endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "llm_model": LLM_MODEL, "index_sizes": app.state.index_sizes}

@app.post("/classify", response_model=ClassifyOut)
def classify_endpoint(body: ComplaintIn):
    clean, redactions = redact(body.text)
    r, used, degraded = run_classifier(clean, body.method)
    return ClassifyOut(**to_classify_out(r, used, degraded, redactions))

@app.post("/triage", response_model=TriageOut)
def triage_endpoint(body: ComplaintIn):
    clean, redactions = redact(body.text)
    r, used, degraded = run_classifier(clean, body.method)
    out = TriageOut(**to_classify_out(r, used, degraded, redactions))
    try:
        d = draft_reply(clean, r["product"], r.get("summary") or "")
        out.reply = d.get("reply")
        out.citations_used = d.get("citations_used", [])
        out.sources = [Source(doc=s["doc"], page=s["page"], text=s["text"])
                       for s in d.get("sources", [])]
        out.reply_error = d.get("error")
    except (groq.APIError, RuntimeError) as e:
        out.reply_error = f"Reply drafting unavailable ({type(e).__name__})"
    return out