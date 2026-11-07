import json
from pathlib import Path
import numpy as np

rows = [json.loads(l) for l in Path("results/faithfulness.jsonl").open(encoding="utf-8")]
ok = [r for r in rows if "verdict" in r]
errors = [r for r in rows if "error" in r]
claims = [(r["complaint_id"], c) for r in ok for c in r["verdict"].get("claims", [])]

score = {"yes": 1.0, "partial": 0.5, "no": 0.0}
n_claims = len(claims)
faith = np.mean([score.get(c["supported"], 0) for _, c in claims])
strict = np.mean([c["supported"] == "yes" for _, c in claims])
fully = np.mean([all(c["supported"] == "yes" for c in r["verdict"].get("claims", []))
                 for r in ok])
coverage = np.mean([bool(c.get("cited")) for _, c in claims])
cited_total = sum(len(c.get("cited", [])) for _, c in claims)
cited_correct = sum(len(set(c.get("cited", [])) & set(c.get("supporting", [])))
                    for _, c in claims)
lat = [r["e2e_latency_s"] for r in ok]

print(f"Replies judged: {len(ok)}   errors: {len(errors)}   "
      f"degraded: {sum(r.get('degraded', False) for r in ok)}")
print(f"Claims audited: {n_claims}   (avg {n_claims / len(ok):.1f} per reply)\n")
print(f"Faithfulness (partial = 0.5):     {faith:.1%}")
print(f"Strict faithfulness (yes only):   {strict:.1%}")
print(f"Fully faithful replies:           {fully:.1%}")
print(f"Citation coverage:                {coverage:.1%}")
print(f"Citation precision:               {cited_correct / max(cited_total, 1):.1%}")
print(f"Invalid citation numbers:         {sum(len(r['invalid_citations']) for r in ok)}")
print(f"Compensation promises:            {sum(r['verdict'].get('promises_compensation', False) for r in ok)}")
print(f"Liability admissions:             {sum(r['verdict'].get('admits_liability', False) for r in ok)}")
print(f"Within 180 words:                 {np.mean([r['words'] <= 180 for r in ok]):.1%} "
      f"(median {int(np.median([r['words'] for r in ok]))} words)")
print(f"End-to-end latency p50 / p95:     {np.median(lat):.1f}s / {np.percentile(lat, 95):.1f}s")

print("\n=== Claims NOT fully supported (review these) ===")
for cid, c in claims:
    if c["supported"] != "yes":
        print(f"[{cid}] {c['supported'].upper()} | cited {c.get('cited')} | "
              f"supporting {c.get('supporting')}\n   {c['claim']}\n")
for r in errors:
    print(f"ERROR [{r['complaint_id']}]: {r['error']}")