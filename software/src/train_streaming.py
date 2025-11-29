#!/usr/bin/env python3
"""
Streaming / incremental trainer (polynomial dataset).

Two modes:
- from features parquet: reads in chunks and uses partial_fit on an SGDRegressor
- directly from data/db: loads files in batches (sage -python) and partial_fit (extracting polynomial features)

Run:
- From parquet:
    python software/src/train_streaming.py --from-parquet dataset/features.parquet --model models/sgd_model.joblib --batch-size 10000
- From DB (requires Sage):
    sage -python software/src/train_streaming.py --from-db --db-dir data/db --model models/sgd_model.joblib --batch-size-files 8
"""
import argparse
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from software.src.features import poly_to_features

def train_from_parquet(parquet_path, model_path, batch_size=10000, target_col="target"):
    scaler = StandardScaler()
    model = SGDRegressor(max_iter=1, tol=None, penalty="l2", random_state=42)
    first = True
    df = pd.read_parquet(parquet_path, engine="pyarrow")
    cols = [c for c in df.columns if c != target_col]
    n = len(df)
    for start in range(0, n, batch_size):
        chunk = df.iloc[start:start+batch_size]
        X = chunk[cols].to_numpy(dtype=float)
        y = chunk[target_col].to_numpy(dtype=float)
        if first:
            scaler.partial_fit(X)
            Xs = scaler.transform(X)
            model.partial_fit(Xs, y)
            first = False
        else:
            scaler.partial_fit(X)
            Xs = scaler.transform(X)
            model.partial_fit(Xs, y)
        print(f"Processed rows {start}:{start+len(chunk)}")
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": cols}, model_path)
    print("Saved incremental model to", model_path)

def train_from_db(db_dir, model_path, batch_size_files=8, target_col="target"):
    from sage.all import load
    scaler = StandardScaler()
    model = SGDRegressor(max_iter=1, tol=None, penalty="l2", random_state=42)
    first = True
    files = sorted(os.listdir(db_dir))
    batch_rows = []
    cols = None
    for fn in files:
        path = os.path.join(db_dir, fn)
        data = load(path)
        if not isinstance(data, dict):
            continue
        for k, v in data.items():
            # expect v to be (graph, poly) or poly
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                poly = v[1]
            else:
                poly = v
            poly_str = str(poly)
            feats = poly_to_features(poly_str, max_deg_cap=6)
            feats["target"] = int(len(k))
            batch_rows.append(feats)
        if len(batch_rows) >= 1000:
            df = pd.DataFrame(batch_rows)
            if cols is None:
                cols = [c for c in df.columns if c != "target"]
            X = df[cols].to_numpy(dtype=float)
            y = df["target"].to_numpy(dtype=float)
            if first:
                scaler.partial_fit(X)
                Xs = scaler.transform(X)
                model.partial_fit(Xs, y)
                first = False
            else:
                scaler.partial_fit(X)
                Xs = scaler.transform(X)
                model.partial_fit(Xs, y)
            print("Trained on batch size", len(df))
            batch_rows = []
    # final flush
    if batch_rows:
        df = pd.DataFrame(batch_rows)
        if cols is None:
            cols = [c for c in df.columns if c != "target"]
        X = df[cols].to_numpy(dtype=float)
        y = df["target"].to_numpy(dtype=float)
        scaler.partial_fit(X)
        Xs = scaler.transform(X)
        model.partial_fit(Xs, y)
    joblib.dump({"model": model, "scaler": scaler, "feature_columns": cols}, model_path)
    print("Saved incremental model to", model_path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-parquet", dest="parquet", default=None)
    p.add_argument("--from-db", action="store_true")
    p.add_argument("--db-dir", default="data/db")
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=10000)
    p.add_argument("--batch-size-files", type=int, default=8)
    args = p.parse_args()

    if args.parquet:
        train_from_parquet(args.parquet, args.model, batch_size=args.batch_size)
    elif args.from_db:
        train_from_db(args.db_dir, args.model, batch_size_files=args.batch_size_files)
    else:
        raise SystemExit("Specify --from-parquet or --from-db")

if __name__ == "__main__":
    main()
