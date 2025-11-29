#!/usr/bin/env python3
"""
Build a dataset of polynomials -> counts from files created with Sage's save().

Each file should contain a top-level dict mapping keys to values, where each
value is a tuple/list (graph, polynomial). We ignore the graph and save the
polynomial as a string along with the target (the length of the key).

Output: a pickle file that is a list of dicts:
  {'polynomial': <str>, 'target': int, 'source_file': str}
This file is intended for downstream numeric feature extraction (features.py).
"""
import os
import argparse
import pickle
from typing import Any, List
from sage.all import load  # requires Sage

def load_file(path: str) -> Any:
    return load(path)

def extract_examples_from_file(path: str) -> List[dict]:
    data = load_file(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"file {path} does not contain a top-level dict (got {type(data)})")
    examples = []
    for k, v in data.items():
        try:
            target = len(k)
        except Exception:
            raise RuntimeError(f"Could not compute len(key) for key: {repr(k)[:200]}")
        # expect v to be (graph, poly) or [graph, poly]
        poly = None
        if isinstance(v, (list, tuple)) and len(v) >= 2:
            poly = v[1]
        else:
            # if the value is just a polynomial, accept it
            poly = v
        # store polynomial as string so downstream can parse with sympy
        examples.append({"polynomial": str(poly), "target": int(target)})
    return examples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", default="data/db")
    p.add_argument("--out", required=True, help="output pickle containing list of {'polynomial':str,'target':int}")
    args = p.parse_args()

    if not os.path.isdir(args.db_dir):
        raise SystemExit("db-dir not found: " + args.db_dir)
    files = sorted(os.listdir(args.db_dir))
    all_examples = []
    for fn in files:
        fp = os.path.join(args.db_dir, fn)
        print("loading", fp)
        exs = extract_examples_from_file(fp)
        for ex in exs:
            ex["source_file"] = fn
        all_examples.extend(exs)

    with open(args.out, "wb") as f:
        pickle.dump(all_examples, f)
    print("wrote", len(all_examples), "examples to", args.out)

if __name__ == "__main__":
    main()
