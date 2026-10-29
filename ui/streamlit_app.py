import os
import httpx
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
LOW_CONFIDENCE = 0.6

SAMPLES = {
    "ATM double charge": (
        "My debit card was charged twice for the same ATM withdrawal of Rs 2,500 "
        "on 3 October. I called customer care twice and emailed them, but nobody "
        "has replied for over a month. Please refund the duplicate debit."
    ),
    "Credit card interest": (
        "I paid my full credit card bill before the due date, but this month's "
        "statement still shows interest and a late payment fee. The bank says the "
        "payment was credited late, but my bank account shows it was debited on time."
    ),
    "Recovery agent calls": (
        "A recovery agent keeps calling me and my family several times a day about a "
        "personal loan that I already closed last year. I have the closure letter. "
        "They threatened to report me to the credit bureau if I don't pay."
    ),
}

st.set_page_config(page_title="ComplaintIQ", page_icon="🏦", layout="wide")

# ---------- API helpers ----------
@st.cache_data(ttl=60)
def get_health():
    try:
        r = httpx.get(f"{API_URL}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPError:
        return None

def call_triage(text: str, method: str) -> dict:
    r = httpx.post(f"{API_URL}/triage", json={"text": text, "method": method},
                   timeout=120)
    if r.status_code == 422:
        raise ValueError("Complaint must be between 20 and 10,000 characters.")
    r.raise_for_status()
    return r.json()

def load_sample(name: str):
    st.session_state.text = SAMPLES[name]

# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")
    method = st.selectbox(
        "Classification method", ["rag", "zero_shot", "knn"],
        help="RAG: best accuracy (0.786 macro-F1). kNN: instant, no LLM.")
    health = get_health()
    if health:
        st.success(f"API online · {health['llm_model']}")
        st.caption(f"Indexed: {health['index_sizes']['complaints']:,} complaints, "
                   f"{health['index_sizes']['policies']} policy chunks")
    else:
        st.error(f"API offline at {API_URL}")
    st.header("Try a sample")
    for name in SAMPLES:
        st.button(name, on_click=load_sample, args=(name,), use_container_width=True)

# ---------- main ----------
st.title("🏦 ComplaintIQ")
st.write("AI triage for banking complaints: classifies the complaint, then drafts a "
         "reply grounded in the RBI Integrated Ombudsman Scheme 2026, with citations.")

if "text" not in st.session_state:
    st.session_state.text = ""
st.text_area("Customer complaint", key="text", height=160,
             placeholder="Paste a customer complaint here...")

if st.button("Analyze complaint", type="primary"):
    if not health:
        st.error("The API is offline. Start it with: python -m uvicorn app.api:app")
        st.stop()
    try:
        with st.spinner("Classifying and drafting a reply..."):
            res = call_triage(st.session_state.text, method)
    except ValueError as e:
        st.warning(str(e))
        st.stop()
    except httpx.HTTPError as e:
        st.error(f"API request failed: {type(e).__name__}")
        st.stop()

    # --- classification ---
    st.markdown(f"#### Category: {res['product']}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Severity", (res.get("severity") or "n/a").title())
    c2.metric("Method used", res["method_used"])
    c3.metric("Classification time", f"{res['latency_s']}s" if res.get("latency_s") else "n/a")

    if res.get("degraded"):
        st.warning("The AI model was unavailable, so a fast fallback classifier (kNN) "
                   "was used. Reply drafting may be unavailable.")
    conf = res.get("knn_confidence")
    if conf is not None and conf < LOW_CONFIDENCE:
        st.warning(f"Low confidence ({conf:.0%} of similar complaints agree). "
                   "Route to a human agent for review.")
    if res.get("summary"):
        st.info(f"**Summary:** {res['summary']}")
    if res.get("redactions"):
        masked = ", ".join(f"{n} {tag.strip('[]').lower()}" for tag, n in res["redactions"].items())
        st.caption(f"🔒 Masked before AI processing: {masked}")

    # --- reply ---
    st.subheader("Drafted reply")
    if res.get("reply"):
        st.text_area("Review and edit before sending", value=res["reply"], height=260)
    else:
        st.error(res.get("reply_error") or "No reply could be drafted.")

    # --- sources ---
    if res.get("sources"):
        st.subheader("RBI sources")
        used = set(res.get("citations_used", []))
        for i, s in enumerate(res["sources"], start=1):
            tag = "✅ cited" if i in used else "not cited"
            with st.expander(f"[{i}] {s['doc']}, page {s['page']} · {tag}"):
                st.write(s["text"])

st.divider()
st.caption("Demo built on public CFPB complaint data and RBI RB-IOS 2026 documents. "
           "Not legal advice. All drafted replies must be reviewed by a human before sending.")