import groq
from app.classifier import classify, classify_rag, classify_knn
from app.drafter import draft_reply
from app.pii import redact

METHODS = {"rag": classify_rag, "zero_shot": classify, "knn": classify_knn}

def run_classifier(text: str, method: str) -> tuple[dict, str, bool]:
    """Returns (result, method_actually_used, degraded)."""
    try:
        result = METHODS[method](text)
        if result.get("product"):
            return result, method, False
    except (groq.APIError, RuntimeError):
        pass   # rate limit, outage, missing key... fall back to kNN
    return classify_knn(text), "knn", method != "knn"

def triage(text: str, method: str = "rag", draft: bool = True) -> dict:
    clean, redactions = redact(text)
    r, used, degraded = run_classifier(clean, method)
    out = {
        "product": r["product"], "severity": r.get("severity"),
        "summary": r.get("summary"), "method_used": used, "degraded": degraded,
        "knn_confidence": r.get("knn_confidence"), "redactions": redactions,
        "latency_s": r.get("latency_s"),
        "reply": None, "citations_used": [], "sources": [], "reply_error": None,
    }
    if draft:
        try:
            d = draft_reply(clean, r["product"], r.get("summary") or "")
            out.update(
                reply=d.get("reply"),
                citations_used=d.get("citations_used", []),
                sources=[{"doc": s["doc"], "page": s["page"], "text": s["text"]}
                         for s in d.get("sources", [])],
                reply_error=d.get("error"),
            )
        except (groq.APIError, RuntimeError) as e:
            out["reply_error"] = f"Reply drafting unavailable ({type(e).__name__})"
    return out