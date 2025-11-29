#!/usr/bin/env python3
"""
Parallel extraction of structured polynomial examples into a single pickle.

Usage:
  sage -python software/src/extract_features_parallel.py --db-dir data/db --out dataset/polys.pkl --n-jobs 8
"""
import os
import argparse
import pickle
from joblib import Parallel, delayed

from software.src.build_dataset import extract_examples_from_file

def process_file(fn, db_dir):
    path = os.path.join(db_dir, fn)
    try:
        exs = extract_examples_from_file(path)
        for ex in exs:
            ex["source_file"] = fn
        return exs
    except Exception as e:
        print("error processing", fn, ":", e)
        return []

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n-jobs", type=int, default=4)
    args = p.parse_args()

    files = sorted(os.listdir(args.db_dir))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(process_file)(fn, args.db_dir) for fn in files
    )

    all_examples = []
    for res in results:
        all_examples.extend(res)

    with open(args.out, "wb") as f:
        pickle.dump(all_examples, f)
    print("Wrote consolidated raw polynomial examples to", args.out, "count:", len(all_examples))

if __name__ == "__main__":
    main()