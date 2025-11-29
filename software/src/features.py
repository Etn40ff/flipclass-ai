#!/usr/bin/env python3
"""
Optional: vectorize polys.pkl into dense arrays up to max-degree (backward compatible).
If you prefer the sparse infinite-index model, you can skip this step and use the
online sparse trainer directly on polys.pkl.

Produces: dataset/features.pkl with keys:
  {"X": ndarray, "y": ndarray, "meta": [...], "monomials": [(i,j)...], "max_degree": D}
"""
import argparse
import pickle
import numpy as np
from typing import List, Tuple, Dict

def parse_coef_keys(coef_map: Dict[str, float]):
    exps = []
    for k in coef_map.keys():
        try:
            i_s, j_s = k.split("_")
            exps.append((int(i_s), int(j_s)))
        except Exception:
            continue
    return exps

def build_monomials(max_degree: int) -> List[Tuple[int,int]]:
    out = []
    for d in range(0, max_degree+1):
        for i in range(d, -1, -1):
            j = d - i
            out.append((i,j))
    return out

def coefs_to_vector(coefs: Dict[str, float], monomials: List[Tuple[int,int]]):
    v = np.zeros(len(monomials), dtype=float)
    for idx, (i,j) in enumerate(monomials):
        v[idx] = float(coefs.get(f"{i}_{j}", 0.0))
    return v

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    p.add_argument("--max-degree", dest="max_degree", type=int, default=None)
    args = p.parse_args()

    with open(args.infile, "rb") as f:
        examples = pickle.load(f)
    if not isinstance(examples, list):
        raise SystemExit("expected list")

    # detect max degree if not provided
    observed = 0
    for ex in examples:
        coefs = ex.get("poly_coefs", {}) or {}
        for (i,j) in parse_coef_keys(coefs):
            observed = max(observed, i + j)
    max_deg = args.max_degree if args.max_degree is not None else observed
    monomials = build_monomials(max_deg)

    X = np.zeros((len(examples), len(monomials)), dtype=float)
    y = np.zeros((len(examples),), dtype=int)
    meta = []
    for k, ex in enumerate(examples):
        X[k, :] = coefs_to_vector(ex.get("poly_coefs", {}) or {}, monomials)
        y[k] = int(ex.get("target", 0))
        meta.append({"source_file": ex.get("source_file")})
    out = {"X": X, "y": y, "meta": meta, "monomials": monomials, "max_degree": max_deg}
    with open(args.outfile, "wb") as f:
        pickle.dump(out, f)
    print("Wrote features:", X.shape, "max_degree:", max_deg)

if __name__ == "__main__":
    main()