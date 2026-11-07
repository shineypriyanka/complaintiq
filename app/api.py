from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI
from pydantic import BaseModel, Field
from app.config import LLM_MODEL
from app.indexing import ensure_indexes
from app.retriever import embed
from app.service import triage

@asynccontextmanager
async def lifespan(app: FastAPI):
    embed(["warm up"])                       # load the model once, at startup
    app.state.index_sizes = ensure_indexes() # build indexes if missing
    yield

app = FastAPI(title="ComplaintIQ API", version="0.2.0", lifespan=lifespan)

class ComplaintIn(BaseModel):
    text: str = Field(..., min_length=20, max_length=10_000)
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

@app.get("/health")
def health():
    return {"status": "ok", "llm_model": LLM_MODEL, "index_sizes": app.state.index_sizes}

@app.post("/classify", response_model=ClassifyOut)
def classify_endpoint(body: ComplaintIn):
    return ClassifyOut(**triage(body.text, body.method, draft=False))

@app.post("/triage", response_model=TriageOut)
def triage_endpoint(body: ComplaintIn):
    return TriageOut(**triage(body.text, body.method))