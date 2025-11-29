#!/usr/bin/env python3
"""
Parallel feature extractor (updated for polynomial dataset).

- Each DB file contains a dict mapping keys -> values where value is (graph, polynomial).
- We ignore the graph and extract numeric features from polynomial strings in parallel
  (one shard per DB file) and produce a consolidated parquet.
Run:
  sage -python -m software.src.extract_features_parallel --db-dir data/db --out dataset/features.parquet --n-jobs 16
"""
import os
import argparse
from joblib import Parallel, delayed
import pandas as pd

from software.src.features import poly_to_features  # reuse feature logic

def load_sage_file(path):
    from sage.all import load
    return load(path)

def process_file(fn, db_dir, out_shard_dir):
    path = os.path.join(db_dir, fn)
    data = load_sage_file(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"{path} did not load to a dict")
    rows = []
    for k, v in data.items():
        target = len(k)
        # value is (graph, poly) or poly alone
        poly = None
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            poly = v[1]
        else:
            poly = v
        poly_str = str(poly)
        feats = poly_to_features(poly_str, max_deg_cap=6)
        feats["target"] = int(target)
        rows.append(feats)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    shard_path = os.path.join(out_shard_dir, fn + ".parquet")
    df.to_parquet(shard_path, index=False)
    return shard_path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", default="data/db")
    p.add_argument("--out", default="dataset/features.parquet", help="final parquet (will be overwritten)")
    p.add_argument("--shard-dir", default="dataset/shards")
    p.add_argument("--n-jobs", type=int, default=8)
    args = p.parse_args()

    os.makedirs(args.shard_dir, exist_ok=True)
    files = sorted(os.listdir(args.db_dir))

    Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(fn, args.db_dir, args.shard_dir) for fn in files
    )

    shard_files = sorted([os.path.join(args.shard_dir, f) for f in os.listdir(args.shard_dir) if f.endswith(".parquet")])
    if not shard_files:
        raise SystemExit("No shards produced")
    out_parts = []
    for sf in shard_files:
        df = pd.read_parquet(sf)
        out_parts.append(df)
    full = pd.concat(out_parts, ignore_index=True)
    full.to_parquet(args.out, index=False)
    print("Wrote consolidated features to", args.out)
    print("Rows:", len(full))

if __name__ == "__main__":
    main()
