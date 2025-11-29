#!/usr/bin/env python3
"""
Parallel feature extractor.
- Reads all files under data/db (Sage .sobj saved with save()).
- For each file, loads examples (using Sage.load), converts graphs to networkx,
  computes features, and writes a per-file parquet shard to out_dir/shards/.
- After all shards are created, optionally merges them into a single features.parquet.

Run (from repo root):
  sage -python -m software.src.extract_features_parallel --db-dir data/db --out dataset/features.parquet --n-jobs 16

Notes:
- Requires pandas pyarrow for parquet writing (pip install pyarrow).
- Shards are safe to remove after concatenation.
"""
import os
import argparse
from pathlib import Path
from joblib import Parallel, delayed
import pandas as pd
import numpy as np
import networkx as nx

# minimal converter (compatible with Sage graph objects saved via save())
def to_networkx_graph(obj):
    if isinstance(obj, nx.DiGraph):
        return obj.copy()
    if isinstance(obj, dict):
        if "nodes" in obj and "edges" in obj:
            G = nx.DiGraph()
            G.add_nodes_from(obj["nodes"])
            G.add_edges_from(obj["edges"])
            return G
        if all(isinstance(v, (list, tuple)) for v in obj.values()):
            G = nx.DiGraph()
            for u, vs in obj.items():
                G.add_node(u)
                for v in vs:
                    G.add_edge(u, v)
            return G
    if hasattr(obj, "to_networkx"):
        nxg = obj.to_networkx()
        if not isinstance(nxg, nx.DiGraph):
            nxg = nx.DiGraph(nxg)
        return nxg
    if hasattr(obj, "vertices") and hasattr(obj, "edges"):
        nodes = list(obj.vertices())
        raw_edges = list(obj.edges())
        edges = []
        for e in raw_edges:
            try:
                if len(e) >= 2:
                    edges.append((e[0], e[1]))
            except Exception:
                try:
                    tup = tuple(e)
                    if len(tup) >= 2:
                        edges.append((tup[0], tup[1]))
                except Exception:
                    pass
        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G
    raise RuntimeError(f"Unsupported graph object of type {type(obj)}")

def features_from_graph(G):
    feats = {}
    feats["n_nodes"] = float(G.number_of_nodes())
    feats["n_edges"] = float(G.number_of_edges())
    indeg = [d for _, d in G.in_degree()]
    outdeg = [d for _, d in G.out_degree()]
    for name, seq in (("indeg", indeg), ("outdeg", outdeg)):
        feats[f"{name}_min"] = float(min(seq)) if seq else 0.0
        feats[f"{name}_max"] = float(max(seq)) if seq else 0.0
        feats[f"{name}_mean"] = float(np.mean(seq)) if seq else 0.0
        feats[f"{name}_std"] = float(np.std(seq)) if seq else 0.0
    feats["n_sources"] = float(sum(1 for v in G.nodes() if G.in_degree(v) == 0))
    feats["n_sinks"] = float(sum(1 for v in G.nodes() if G.out_degree(v) == 0))
    def topological_level_counts(G, max_levels=6):
        if not nx.is_directed_acyclic_graph(G):
            return [0] * max_levels
        level = {}
        for v in nx.topological_sort(G):
            if G.in_degree(v) == 0:
                level[v] = 0
            else:
                level[v] = 1 + max(level[p] for p in G.predecessors(v))
        counts = [0] * max_levels
        for v, l in level.items():
            if l < max_levels:
                counts[l] += 1
        return counts
    levels = topological_level_counts(G, max_levels=6)
    for i, c in enumerate(levels):
        feats[f"level_count_{i}"] = float(c)
    def count_st_paths_dag(G):
        if not nx.is_directed_acyclic_graph(G):
            return -1.0
        topo = list(nx.topological_sort(G))
        dp = {v: 0 for v in topo}
        sources = [v for v in G.nodes() if G.in_degree(v) == 0]
        sinks = [v for v in G.nodes() if G.out_degree(v) == 0]
        for s in sources:
            dp[s] = dp.get(s, 0) + 1
        for v in topo:
            for w in G.successors(v):
                dp[w] = dp.get(w, 0) + dp[v]
        return float(sum(dp[t] for t in sinks))
    feats["unlabelled_st_paths"] = float(count_st_paths_dag(G))
    return feats

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
        G = to_networkx_graph(v)
        feats = features_from_graph(G)
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

    results = Parallel(n_jobs=args.n_jobs)(
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
