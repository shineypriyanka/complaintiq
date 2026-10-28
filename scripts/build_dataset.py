from pathlib import Path
import pandas as pd

RAW = Path("data/raw/complaints_raw.parquet")
OUT_DIR = Path("data/processed")
TEXT, LABEL = "consumer_complaint_narrative", "product"

MIN_DATE = "2023-01-01"   # recent = current category names
MIN_WORDS = 20            # too short = no signal
MIN_PER_CLASS = 100       # drop classes too small to evaluate fairly
TEST_PER_CLASS = 40       # fixed, balanced test set
TRAIN_CAP = 500           # max per class in the few-shot/RAG pool
SEED = 42

def step(df, msg):
    print(f"{msg:<40} {len(df):>8,} rows")
    return df

def main():
    df = step(pd.read_parquet(RAW), "Loaded raw")
    df = df[df[TEXT].notna() & (df[TEXT].str.strip() != "")]
    df = step(df, "Has narrative")
    df = df[df["date_received"] >= pd.Timestamp(MIN_DATE, tz="UTC")]
    df = step(df, f"Received since {MIN_DATE}")
    df = df.drop_duplicates(subset=TEXT)
    df = step(df, "Duplicate narratives removed")
    df = df[df[TEXT].str.split().str.len() >= MIN_WORDS]
    df = step(df, f"At least {MIN_WORDS} words")

    counts = df[LABEL].value_counts()
    keep = counts[counts >= MIN_PER_CLASS].index
    print("\nDropped (too few rows):", dict(counts[counts < MIN_PER_CLASS]))
    df = step(df[df[LABEL].isin(keep)], "Classes with enough data")

    test = df.groupby(LABEL).sample(n=TEST_PER_CLASS, random_state=SEED)
    rest = df.drop(test.index)
    train = rest.groupby(LABEL).sample(frac=1, random_state=SEED) \
                .groupby(LABEL).head(TRAIN_CAP)

    assert set(test.index).isdisjoint(train.index), "Leakage between test and train!"

    cols = ["complaint_id", "date_received", TEXT, LABEL, "issue", "company"]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test[cols].to_parquet(OUT_DIR / "test.parquet", index=False)
    train[cols].to_parquet(OUT_DIR / "train.parquet", index=False)

    print("\nPer-class counts:")
    print(pd.DataFrame({"test": test[LABEL].value_counts(),
                        "train": train[LABEL].value_counts()}))

if __name__ == "__main__":
    main()