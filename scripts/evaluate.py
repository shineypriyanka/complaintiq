import argparse, json, time
from collections import Counter
from pathlib import Path
import groq
import pandas as pd
from app.config import LLM_MODEL
from app.classifier import classify, classify_rag, truncate
from app.retriever import retrieve

TEXT, LABEL = "consumer_complaint_narrative", "product"
PER_CLASS, SEED = 20, 42
TPM_LIMIT, SAFETY = 8000, 1.15   # stay safely under tokens-per-minute

def eval_set() -> pd.DataFrame:
    test = pd.read_parquet("data/processed/test.parquet")
    return test.groupby(LABEL).sample(n=PER_CLASS, random_state=SEED)

def knn(text: str, k: int = 5) -> str:
    labels = [e["product"] for e in retrieve(truncate(text), k=k)]
    return Counter(labels).most_common(1)[0][0]   # ties -> nearest neighbour wins

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["knn", "zero_shot", "rag"], required=True)
    method = p.parse_args().method

    model = "none" if method == "knn" else LLM_MODEL.replace("/", "_")
    out = Path(f"results/{method}__{model}.jsonl")
    out.parent.mkdir(exist_ok=True)
    done = set()
    if out.exists():
        done = {json.loads(line)["complaint_id"] for line in out.open(encoding="utf-8")}

    data = eval_set()
    todo = data[~data["complaint_id"].astype(str).isin(done)]
    print(f"{method} | model={model} | done={len(done)} | remaining={len(todo)}")

    if method != "zero_shot":
        retrieve("warm up", k=1)   # load model + DB first, so cold start isn't timed

    with out.open("a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            t0 = time.perf_counter()
            try:
                if method == "knn":
                    r = {"product": knn(row[TEXT]), "valid": True,
                         "latency_s": round(time.perf_counter() - t0, 3)}
                elif method == "zero_shot":
                    r = classify(row[TEXT])
                else:
                    r = classify_rag(row[TEXT])
            except groq.RateLimitError as e:
                print(f"\nRate limit still hit after retries (likely the DAILY cap).")
                print(f"Progress saved. Rerun the same command later to resume.\n{str(e)[:200]}")
                break

            rec = {"complaint_id": str(row["complaint_id"]), "true": row[LABEL],
                   "pred": r.get("product"), "valid": r.get("valid"),
                   "latency_s": r.get("latency_s"),
                   "prompt_tokens": r.get("prompt_tokens", 0),
                   "completion_tokens": r.get("completion_tokens", 0),
                   "error": r.get("error")}
            f.write(json.dumps(rec) + "\n")
            f.flush()   # saved immediately: a crash loses at most one row

            if i % 10 == 0:
                print(f"  {i}/{len(todo)}")
            tokens = rec["prompt_tokens"] + rec["completion_tokens"]
            if tokens:
                time.sleep(tokens / TPM_LIMIT * 60 * SAFETY)

if __name__ == "__main__":
    main()