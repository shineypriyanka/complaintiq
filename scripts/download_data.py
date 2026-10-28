"""Download a few shards of the CFPB Consumer Complaint Database.
Source: Hugging Face mirror of the official CFPB CSV (CC0 data).
https://huggingface.co/datasets/Mouwiya/cfpb-consumer-complaints
"""
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files

REPO = "Mouwiya/cfpb-consumer-complaints"
N_SHARDS = 3   # ~20 MB; increase later if we need more rows
OUT = Path("data/raw/complaints_raw.parquet")

def main():
    files = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("data/") and f.endswith(".parquet"))
    print(f"Total shards available: {len(files)}")

    chosen = files[:N_SHARDS]
    frames = []
    for f in chosen:
        path = hf_hub_download(REPO, f, repo_type="dataset")
        frames.append(pd.read_parquet(path))
        print(f"Loaded {f}: {len(frames[-1]):,} rows")

    df = pd.concat(frames, ignore_index=True)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT, index=False)
    print(f"Saved {OUT}: {len(df):,} rows")

if __name__ == "__main__":
    main()