#!/usr/bin/env python3
"""
Feature extraction from polynomials.

Input: pickle created by build_dataset.py (list of {'polynomial':str, 'target':int, ...})
Output: pickle {"X": pandas.DataFrame, "y": np.ndarray, "meta": [...]} 

Features derived from polynomial (string) using sympy when available, else a simple parser:
- total_degree, degree_x, degree_y, n_terms
- coeff_min/coeff_max/coeff_mean/coeff_std
- coeff_l1 / coeff_l2
- counts of monomials by total degree up to max_deg_cap (caps at 6)
"""
import argparse
import pickle
from typing import Dict
import pandas as pd
import numpy as np

def poly_to_features(poly_str: str, max_deg_cap: int = 6) -> Dict[str, float]:
    s = poly_str.replace("^", "**")
    coeffs = []
    degrees = []
    deg_x = 0
    deg_y = 0
    n_terms = 0
    try:
        import sympy as sp
        x, y = sp.symbols("x y")
        expr = sp.sympify(s)
        P = sp.Poly(expr, x, y)
        terms = P.terms()  # list of ((i,j), coeff)
        for (i, j), c in terms:
            try:
                cf = float(sp.N(c))
            except Exception:
                cf = float(c)
            coeffs.append(cf)
            degrees.append(i + j)
            deg_x = max(deg_x, i)
            deg_y = max(deg_y, j)
            n_terms += 1
    except Exception:
        # fallback: crude parser: find monomials like coef*x^i*y^j or x^i*y^j or numeric constants
        import re
        # replace '-' with '+-' to split terms
        t = s.replace("-", "+-")
        parts = [p.strip() for p in t.split("+") if p.strip() != ""]
        for p in parts:
            # attempt to extract coefficient
            coef = 1.0
            px = p
            m = re.match(r"^([+-]?[0-9]*(?:\.[0-9]+)?)\s*\*?\s*(.*)$", p)
            if m:
                coef_s, rest = m.groups()
                if coef_s not in ("", "+", "-"):
                    try:
                        coef = float(coef_s)
                        px = rest
                    except Exception:
                        coef = 1.0
                        px = p
                else:
                    if coef_s == "-":
                        coef = -1.0
                    px = rest
            # find exponents
            ix = 0
            iy = 0
            mx = re.search(r"x(?:\s*\*\*|\^)\s*(\d+)", px)
            my = re.search(r"y(?:\s*\*\*|\^)\s*(\d+)", px)
            if mx:
                ix = int(mx.group(1))
            elif "x" in px:
                ix = 1
            if my:
                iy = int(my.group(1))
            elif "y" in px:
                iy = 1
            coeffs.append(coef)
            degrees.append(ix + iy)
            deg_x = max(deg_x, ix)
            deg_y = max(deg_y, iy)
            n_terms += 1

    # compute numeric summaries
    if coeffs:
        arr = np.array(coeffs, dtype=float)
        coeff_min = float(np.min(arr))
        coeff_max = float(np.max(arr))
        coeff_mean = float(np.mean(arr))
        coeff_std = float(np.std(arr))
        coeff_l1 = float(np.sum(np.abs(arr)))
        coeff_l2 = float(np.sqrt(np.sum(arr * arr)))
        n_nonzero = int(np.sum(arr != 0))
    else:
        coeff_min = coeff_max = coeff_mean = coeff_std = coeff_l1 = coeff_l2 = 0.0
        n_nonzero = 0

    # monomial degree counts up to cap
    degree_counts = {f"deg_count_{d}": 0.0 for d in range(max_deg_cap + 1)}
    for d in degrees:
        if d <= max_deg_cap:
            degree_counts[f"deg_count_{d}"] += 1.0

    feats = {
        "total_degree": float(max(degrees) if degrees else 0),
        "degree_x": float(deg_x),
        "degree_y": float(deg_y),
        "n_terms": float(n_terms),
        "coeff_min": coeff_min,
        "coeff_max": coeff_max,
        "coeff_mean": coeff_mean,
        "coeff_std": coeff_std,
        "coeff_l1": coeff_l1,
        "coeff_l2": coeff_l2,
        "n_nonzero_coeffs": float(n_nonzero),
    }
    feats.update(degree_counts)
    return feats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()

    with open(args.infile, "rb") as f:
        examples = pickle.load(f)

    rows = []
    y = []
    meta = []
    for item in examples:
        poly_str = item.get("polynomial") or str(item)
        feats = poly_to_features(poly_str, max_deg_cap=6)
        rows.append(feats)
        y.append(int(item.get("target", 0)))
        meta.append({"source_file": item.get("source_file")})
    X = pd.DataFrame(rows)
    y_arr = np.array(y, dtype=int)
    with open(args.out, "wb") as f:
        pickle.dump({"X": X, "y": y_arr, "meta": meta}, f)
    print("Wrote features:", X.shape)

if __name__ == "__main__":
    main()
