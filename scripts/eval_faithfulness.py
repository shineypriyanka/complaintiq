import json
import time
from pathlib import Path
import groq
import pandas as pd
from app.service import triage
from app.judge import judge_reply

TEXT, LABEL = "consumer_complaint_narrative", "product"
N_PER_CLASS, SEED = 2, 7
TPM_LIMIT, SAFETY, GEN_TOKENS_EST = 8000, 1.15, 3500
OUT = Path("results/faithfulness.jsonl")

def sample() -> pd.DataFrame:
    test = pd.read_parquet("data/processed/test.parquet")
    return test.groupby(LABEL).sample(n=N_PER_CLASS, random_state=SEED)

def main():
    done = set()
    if OUT.exists():
        done = {json.loads(l)["complaint_id"] for l in OUT.open(encoding="utf-8")}
    data = sample()
    todo = data[~data["complaint_id"].astype(str).isin(done)]
    print(f"faithfulness | done={len(done)} | remaining={len(todo)}")

    with OUT.open("a", encoding="utf-8") as f:
        for i, (_, row) in enumerate(todo.iterrows(), 1):
            rec = {"complaint_id": str(row["complaint_id"]), "true": row[LABEL]}
            judge_tokens = 0
            try:
                t0 = time.perf_counter()
                out = triage(row[TEXT], "rag")
                rec["e2e_latency_s"] = round(time.perf_counter() - t0, 2)
                rec.update(pred=out["product"], degraded=out["degraded"],
                           reply=out["reply"],
                           sources=[{"n": n, "doc": s["doc"], "page": s["page"],
                                     "text": s["text"]}
                                    for n, s in enumerate(out["sources"], start=1)])
                if out["reply"]:
                    reply = out["reply"]
                    rec["words"] = len(reply.split())
                    rec["invalid_citations"] = [n for n in out["citations_used"]
                                                if not 1 <= n <= len(out["sources"])]
                    verdict, jres = judge_reply(reply, out["sources"])
                    rec["verdict"] = verdict
                    judge_tokens = jres.prompt_tokens + jres.completion_tokens
                else:
                    rec["error"] = out.get("reply_error") or "no reply"
            except groq.RateLimitError as e:
                print(f"\nDaily cap likely hit. Progress saved; rerun later.\n{str(e)[:200]}")
                break
            except (json.JSONDecodeError, groq.BadRequestError) as e:
                rec["error"] = f"{type(e).__name__}: {str(e)[:150]}"

            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"  {i}/{len(todo)}  {rec.get('pred', '?')[:30]}  "
                  f"claims={len(rec.get('verdict', {}).get('claims', []))}")
            time.sleep((GEN_TOKENS_EST + judge_tokens) / TPM_LIMIT * 60 * SAFETY)

if __name__ == "__main__":
    main()