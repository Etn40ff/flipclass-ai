#!/usr/bin/env python3
"""
Quick data inspector for data/db files.

Usage:
  python software/src/check_data.py --db-dir data/db --sample 5
"""

import os
import argparse
import pickle
import json
import ast
from pprint import pprint

def try_load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            return {"__error__": f"pickle load failed: {e}"}
    elif ext in (".json",):
        with open(path, "r") as f:
            return json.load(f)
    else:
        # try text -> python literal
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return ast.literal_eval(text)
        except Exception as e:
            return {"__error__": f"text load failed: {e}"}

def inspect_file(path, sample=3):
    print("===", path)
    data = try_load(path)
    t = type(data)
    print("type:", t)
    if isinstance(data, dict):
        keys = list(data.keys())
        print("num keys:", len(keys))
        for k in keys[:sample]:
            v = data[k]
            print("key-type:", type(k), "value-type:", type(v))
            try:
                print("  key repr:", repr(k)[:200])
            except:
                pass
            try:
                print("  value repr:", repr(v)[:200])
            except:
                pass
    else:
        print("Top-level object (non-dict): repr:", repr(data)[:500])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", default="data/db", help="directory with data files")
    parser.add_argument("--sample", type=int, default=3)
    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        print("db dir not found:", args.db_dir)
        return

    files = sorted(os.listdir(args.db_dir))
    if not files:
        print("no files in", args.db_dir)
        return
    for fn in files[:args.sample]:
        inspect_file(os.path.join(args.db_dir, fn), sample=3)

if __name__ == "__main__":
    main()