import json
import random
from pathlib import Path

rows = [json.loads(l) for l in Path("results/faithfulness.jsonl").open(encoding="utf-8")]
items = []
for r in rows:
    src = {s["n"]: s for s in r.get("sources", [])}
    for c in r.get("verdict", {}).get("claims", []):
        items.append((r["complaint_id"], c, src))

random.seed(0)
yes = [x for x in items if x[1]["supported"] == "yes"]
to_review = [x for x in items if x[1]["supported"] != "yes"] + random.sample(yes, 10)

for k, (cid, c, src) in enumerate(to_review, start=1):
    print("=" * 90)
    print(f"#{k} [{cid}]  JUDGE SAYS: {c['supported'].upper()}  "
          f"| cited {c.get('cited')} | supporting {c.get('supporting')}")
    print(f"CLAIM: {c['claim']}\n")
    for n in sorted(set(c.get("cited", [])) | set(c.get("supporting", []))):
        if n in src:
            print(f"  [{n}] {src[n]['doc']} p.{src[n]['page']}:\n  {src[n]['text']}\n")