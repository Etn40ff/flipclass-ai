#!/usr/bin/env python3
"""
Load DB files and convert stored polynomials to structured coefficient maps.

Each output example is a dict:
  {
    "poly_coefs": {"i_j": coeff, ...},
    "target": int,
    "source_file": str
  }

This script prefers to parse polynomials via sympy when given as strings, but
also accepts objects produced by Sage if you run with sage -python (load()).
"""
import os
import argparse
import pickle
from typing import Any, Dict, List

def poly_to_coefs(poly) -> Dict[str, float]:
    """
    Convert a polynomial (string or sympy/sage object) to a dict "i_j" -> float.
    Uses sympy when available; otherwise falls back to a best-effort text parser.
    """
    # If it's already a mapping, pass through
    if isinstance(poly, dict):
        # assume keys already "i_j" -> coeff
        out = {}
        for k, v in poly.items():
            try:
                out[str(k)] = float(v)
            except Exception:
                try:
                    out[str(k)] = float(str(v))
                except Exception:
                    out[str(k)] = 0.0
        return out

    s = str(poly)
    try:
        import sympy as sp
        x, y = sp.symbols("x y")
        expr = sp.sympify(s)
        P = sp.Poly(expr, x, y)
        terms = P.terms()
        coefs = {}
        for (i, j), c in terms:
            try:
                cf = float(sp.N(c))
            except Exception:
                cf = float(c)
            coefs[f"{i}_{j}"] = cf
        return coefs
    except Exception:
        # Fallback simple parser: split terms, detect x^i y^j
        import re
        t = s.replace("-", "+-")
        parts = [p.strip() for p in t.split("+") if p.strip() != ""]
        coefs = {}
        for p in parts:
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
            key = f"{ix}_{iy}"
            coefs[key] = coefs.get(key, 0.0) + float(coef)
        return coefs

def extract_examples_from_file(path: str) -> List[Dict]:
    """
    Load a DB file and return a list of examples with structured poly maps.
    The loader used depends on file contents:
      - If the file is a pickle of a dict, we try to read it.
      - Otherwise, if Sage load is available and the file contains Sage objects,
        you can call this under sage -python and import load here.
    """
    # Try pickle first
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
    except Exception:
        # try sage load if running under sage environment
        try:
            from sage.all import load  # type: ignore
            data = load(path)
        except Exception as e:
            raise RuntimeError(f"Could not load {path}: {e}")

    if not isinstance(data, dict):
        # If a list/dict of examples already, try to normalize
        # Accept list of tuples (key, (graph, poly)) or similar
        examples = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "poly_coefs" in item:
                    examples.append(item)
            return examples
        raise RuntimeError(f"file {path} does not contain expected top-level dict/list (got {type(data)})")

    out = []
    for key, val in data.items():
        # target heuristic: if key is a sequence, use its length; otherwise 0
        try:
            target = int(len(key))
        except Exception:
            target = 0
        # expect val to be (graph, poly) or similar
        poly = None
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            poly = val[1]
        else:
            poly = val
        coefs = poly_to_coefs(poly)
        out.append({"poly_coefs": coefs, "target": int(target)})
    # annotate source_file at caller level
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", required=True, help="directory with DB files")
    p.add_argument("--out", required=True, help="output pickle path for consolidated examples")
    args = p.parse_args()

    files = sorted(os.listdir(args.db_dir))
    all_examples = []
    for fn in files:
        path = os.path.join(args.db_dir, fn)
        try:
            exs = extract_examples_from_file(path)
        except Exception as e:
            print("skipping", fn, "load error:", e)
            continue
        for ex in exs:
            ex["source_file"] = fn
        all_examples.extend(exs)

    with open(args.out, "wb") as f:
        pickle.dump(all_examples, f)
    print("Wrote", len(all_examples), "examples to", args.out)

if __name__ == "__main__":
    main()