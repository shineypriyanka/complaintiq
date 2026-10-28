from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from app.labels import LABELS

# Groq on-demand prices, USD per 1M tokens (input, output).
# Verify on Groq's pricing page before quoting. Your free tier actually costs $0.
PRICES = {
    "openai_gpt-oss-120b": (0.15, 0.60),
    "openai_gpt-oss-20b": (0.075, 0.30),
    "none": (0.0, 0.0),
}

def load(path: Path):
    df = pd.read_json(path, lines=True, dtype={"complaint_id": str})
    method, model = path.stem.split("__")
    return method, model, df

def macro_f1(y_true, y_pred) -> float:
    return f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)

def bootstrap_ci(y_true, y_pred, n: int = 1000, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    scores = [macro_f1(y_true[s], y_pred[s])
              for s in (rng.choice(idx, len(idx), replace=True) for _ in range(n))]
    return np.percentile(scores, [2.5, 97.5])

def main():
    runs = [load(p) for p in sorted(Path("results").glob("*.jsonl"))]
    common = set.intersection(*(set(df["complaint_id"]) for _, _, df in runs))
    print(f"Comparing {len(runs)} runs on {len(common)} common complaints\n")

    summary, per_class = [], {}
    for method, model, df in runs:
        df = df[df["complaint_id"].isin(common)]
        y_true = df["true"].to_numpy()
        y_pred = df["pred"].fillna("INVALID").to_numpy()
        lo, hi = bootstrap_ci(y_true, y_pred)
        p_in, p_out = PRICES.get(model, (np.nan, np.nan))
        name = f"{method} ({model})"

        summary.append({
            "run": name,
            "accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": macro_f1(y_true, y_pred),
            "f1_95%_CI": f"{lo:.3f}-{hi:.3f}",
            "invalid_%": 100 * (df["pred"].isna()).mean(),
            "p50_lat_s": df["latency_s"].median(),
            "p95_lat_s": df["latency_s"].quantile(0.95),
            "tok_in": df["prompt_tokens"].mean(),
            "tok_out": df["completion_tokens"].mean(),
            "usd_per_1k": (df["prompt_tokens"].mean() * p_in
                           + df["completion_tokens"].mean() * p_out) / 1e6 * 1000,
        })
        per_class[name] = f1_score(y_true, y_pred, labels=LABELS,
                                   average=None, zero_division=0)

        errors = df[df["true"] != df["pred"]]
        print(f"Top confusions, {name}:")
        pairs = (errors["true"].str[:22] + "  ->  " + errors["pred"].fillna("INVALID").str[:22])
        print(pairs.value_counts().head(5).to_string(), "\n")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    table = pd.DataFrame(summary).round(3)
    print("=== SUMMARY ===")
    print(table.to_string(index=False))
    table.to_csv("results/summary.csv", index=False)

    print("\n=== PER-CLASS F1 ===")
    pc = pd.DataFrame(per_class, index=[l[:30] for l in LABELS]).round(2)
    print(pc.to_string())

if __name__ == "__main__":
    main()