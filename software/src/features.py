#!/usr/bin/env python3
"""
Feature extraction.

Reads canonical dataset produced by build_dataset.py and outputs dataset/features.pkl
containing {'X': pandas.DataFrame, 'y': numpy.array, 'meta': list}.
"""

import argparse
import pickle
import networkx as nx
import numpy as np
import pandas as pd
from collections import defaultdict

def count_s_to_t_paths_in_dag(G):
    if not nx.is_directed_acyclic_graph(G):
        return -1
    topo = list(nx.topological_sort(G))
    dp = {v: 0 for v in topo}
    sources = [v for v in G.nodes() if G.in_degree(v) == 0]
    sinks = [v for v in G.nodes() if G.out_degree(v) == 0]
    for s in sources:
        dp[s] = dp.get(s, 0) + 1
    for v in topo:
        for w in G.successors(v):
            dp[w] = dp.get(w, 0) + dp[v]
    total = sum(dp[t] for t in sinks)
    return int(total)

def topological_level_counts(G):
    if not nx.is_directed_acyclic_graph(G):
        return []
    levels = dict()
    for n in nx.topological_sort(G):
        if G.in_degree(n) == 0:
            levels[n] = 0
        else:
            levels[n] = 1 + max(levels[p] for p in G.predecessors(n))
    maxl = max(levels.values()) if levels else 0
    counts = [0]*(maxl+1)
    for v, l in levels.items():
        counts[l] += 1
    return counts

def adjacency_spectrum_features(G, k=6):
    try:
        A = nx.to_numpy_array(G, dtype=float)
        symA = (A + A.T) / 2.0
        vals = np.linalg.eigvals(symA)
        vals = np.sort(np.real(vals))
        if len(vals) > k:
            pick = list(vals[:k//2]) + list(vals[-(k - k//2):])
            arr = np.array(np.round(pick, 6), dtype=float)
        else:
            arr = np.zeros(k, dtype=float)
            arr[:len(vals)] = np.round(vals, 6)
        return arr
    except Exception:
        return np.zeros(k, dtype=float)

def features_from_graph(G):
    feats = {}
    feats['n_nodes'] = G.number_of_nodes()
    feats['n_edges'] = G.number_of_edges()
    indeg = sorted([d for _, d in G.in_degree()])
    outdeg = sorted([d for _, d in G.out_degree()])
    for name, seq in [('indeg', indeg), ('outdeg', outdeg)]:
        feats[f'{name}_min'] = float(seq[0]) if seq else 0.0
        feats[f'{name}_max'] = float(seq[-1]) if seq else 0.0
        feats[f'{name}_mean'] = float(np.mean(seq)) if seq else 0.0
        feats[f'{name}_std'] = float(np.std(seq)) if seq else 0.0
    feats['n_sources'] = sum(1 for v in G.nodes() if G.in_degree(v) == 0)
    feats['n_sinks'] = sum(1 for v in G.nodes() if G.out_degree(v) == 0)
    levels = topological_level_counts(G)
    for i in range(6):
        feats[f'level_count_{i}'] = float(levels[i]) if i < len(levels) else 0.0
    feats['unlabelled_st_paths'] = float(count_s_to_t_paths_in_dag(G))
    spec = adjacency_spectrum_features(G, k=6)
    for i, val in enumerate(spec):
        feats[f'spec_{i}'] = float(val)
    return feats

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True, help="canonical_dataset.pkl")
    parser.add_argument("--out", dest="outfile", required=True, help="output features pickle")
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        canonical_list = pickle.load(f)
    rows = []
    meta = []
    ys = []
    for item in canonical_list:
        G = item['graph']
        y = int(item['count'])
        feats = features_from_graph(G)
        rows.append(feats)
        meta.append({'canonical_id': item.get('canonical_id')})
        ys.append(y)
    X = pd.DataFrame(rows)
    y = np.array(ys, dtype=int)
    out = {'X': X, 'y': y, 'meta': meta}
    with open(args.outfile, "wb") as f:
        pickle.dump(out, f)
    print("Wrote features to", args.outfile)
    print("X shape:", X.shape, "y shape:", y.shape)

if __name__ == "__main__":
    main()