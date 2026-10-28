import pandas as pd

TEXT = "consumer_complaint_narrative"

df = pd.read_parquet("data/raw/complaints_raw.parquet")
print("Shape:", df.shape)
print("Date range:", df["date_received"].min(), "->", df["date_received"].max())

df = df[df[TEXT].notna() & (df[TEXT].str.strip() != "")]
print(f"\nRows with narrative: {len(df):,}")

print("\nProduct distribution (%):")
print((df["product"].value_counts(normalize=True) * 100).round(1))

print("\nTop 15 issues:")
print(df["issue"].value_counts().head(15))

words = df[TEXT].str.split().str.len()
print("\nNarrative length (words):")
print(words.describe().round(0))

print("\nSample complaint:\n", df[TEXT].iloc[0][:600])