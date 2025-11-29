#!/usr/bin/env python3
"""
Streaming / incremental trainer using sparse triangular-index representation.

This trainer reads the raw polynomial examples pickle (polys.pkl) produced by
extract_features_parallel.py (each example contains 'poly_coefs' mapping
i_j' -> coeff and 'target') and trains a SparseSGDRegressor that can grow
to accommodate arbitrarily large monomials (no fixed max-degree).

Example:
  python software/src/train_streaming.py --from-polys dataset/polys.pkl --model models/sparse_model.joblib --eta0 1e-3 --l2 1e-5 --batch-size 2000

Notes:
 - The model stores weights as a dict mapping triangular-index -> coefficient.
 - Optionally normalize each sample by its L2 norm (default True).
"""
import argparse
import pickle
import joblib
from typing import Dict
from software.src.online_model import coef_map_to_indexed_sparse, SparseSGDRegressor
import math

def stream_train(polys_pickle: str, model_path: str, batch_size: int = 2000, eta0: float = 1e-3, l2: float = 1e-5, normalize: bool = True):
    with open(polys_pickle, "rb") as f:
        examples = pickle.load(f)
    if not isinstance(examples, list):
        raise SystemExit("expected list of examples in polys pickle")

    model = SparseSGDRegressor(eta0=eta0, l2=l2, normalize=normalize)

    N = len(examples)
    start = 0
    while start < N:
        end = min(N, start + batch_size)
        batch = examples[start:end]
        X_sparse = []
        y = []
        for ex in batch:
            coefs = ex.get("poly_coefs", {}) or {}
            xsp = coef_map_to_indexed_sparse(coefs)
            X_sparse.append(xsp)
            y.append(float(ex.get("target", 0)))
        model.partial_fit_batch(X_sparse, y, verbose=False)
        mse, cnt = model.score_on_batch(X_sparse, y)
        print(f"Trained on {start}:{end} (batch {cnt}). batch_mse={{mse:.6g}}, total_weights={{len(model.weights)}}")
        start = end

    # Save model
    meta = {
        "model": model.to_dict(),
        "format_version": 1,
    }
    joblib.dump(meta, model_path)
    print("Saved sparse model to", model_path)
    print("Final weights count:", len(model.weights))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-polys", dest="polys", required=True, help="pickle produced by extract_features_parallel.py")
    p.add_argument("--model", required=True, help="output joblib model path")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--eta0", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="disable per-sample L2 normalization")
    args = p.parse_args()

    stream_train(args.polys, args.model, batch_size=args.batch_size, eta0=args.eta0, l2=args.l2, normalize=args.normalize)

if __name__ == "__main__":
    main()