#!/usr/bin/env python3
"""
Inspector for data/db files saved with Sage's save().
Now expects values to be (graph, polynomial) tuples and inspects polynomial info.
Run with Sage's python: sage -python software/src/check_data.py --db-dir data/db
"""
import os
import argparse
from typing import Any
from sage.all import load  # requires Sage
import re

def load_file(path: str) -> Any:
    return load(path)

def poly_summary(poly) -> str:
    try:
        s = str(poly)
        # short, sanitized
        s2 = re.sub(r"\s+", " ", s)
        if len(s2) > 200:
            s2 = s2[:200] + "..."
        return s2
    except Exception:
        return repr(poly)[:200]

def inspect(path: str, sample: int = 3) -> None:
    print("===", path)
    try:
        data = load_file(path)
    except Exception as e:
        print("LOAD ERROR (Sage.load):", e)
        return
    print("top-level type:", type(data))
    if not isinstance(data, dict):
        print("EXPECTED dict at top level; got:", type(data))
        return
    keys = list(data.keys())
    print("num examples in file:", len(keys))
    for k in keys[:sample]:
        v = data[k]
        print(" - key type:", type(k), "len(key):", end=" ")
        try:
            print(len(k))
        except Exception:
            print("N/A")
        print("   value type:", type(v))
        try:
            if isinstance(v, (list, tuple)) and len(v) >= 2:
                poly = v[1]
                print("   poly repr:", poly_summary(poly))
            else:
                print("   value repr:", repr(v)[:200].replace('\n', ' '))
        except Exception as e:
            print("   poly inspect error:", e)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", default="data/db")
    p.add_argument("--sample", type=int, default=3)
    args = p.parse_args()
    if not os.path.isdir(args.db_dir):
        raise SystemExit("db-dir not found: " + args.db_dir)
    files = sorted(os.listdir(args.db_dir))
    if not files:
        raise SystemExit("no files in " + args.db_dir)
    for fn in files[: args.sample]:
        inspect(os.path.join(args.db_dir, fn), sample=args.sample)

if __name__ == "__main__":
    main()
