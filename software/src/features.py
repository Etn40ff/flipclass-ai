#!/usr/bin/env python3
"""
Feature extraction for canonical dataset.

Outputs a pickle with {'X': pandas.DataFrame, 'y': np.ndarray, 'meta': [...]} 
"""
import argparse
import pickle
from typing import Dict
import networkx as nx
import numpy as np
import pandas as pd

def count_st_paths_dag(G: nx.DiGraph) -> int:
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
    return int(sum(dp[t] for t in sinks))

def topological_level_counts(G: nx.DiGraph, max_levels: int = 6):
    if not nx.is_directed_acyclic_graph(G):
        return [0] * max_levels
    level = {}
    for v in nx.topological_sort(G):
        if G.in_degree(v) == 0:
            level[v] = 0
        else:
            level[v] = 1 + max(level[p] for p in G.predecessors(v))
    maxl = max(level.values()) if level else 0
    counts = [0] * max_levels
    for v, l in level.items():
        if l < max_levels:
            counts[l] += 1
    return counts

def features_from_graph(G: nx.DiGraph) -> Dict[str, float]:
    feats = {}
    feats["n_nodes"] = float(G.number_of_nodes())
    feats["n_edges"] = float(G.number_of_edges())
    indeg = [d for _, d in G.in_degree()]
    outdeg = [d for _, d in G.out_degree()]
    for name, seq in ("indeg", indeg), ("outdeg", outdeg):
        feats[f"{name}_min"] = float(min(seq)) if seq else 0.0
        feats[f"{name}_max"] = float(max(seq)) if seq else 0.0
        feats[f"{name}_mean"] = float(np.mean(seq)) if seq else 0.0
        feats[f"{name}_std"] = float(np.std(seq)) if seq else 0.0
    feats["n_sources"] = float(sum(1 for v in G.nodes() if G.in_degree(v) == 0))
    feats["n_sinks"] = float(sum(1 for v in G.nodes() if G.out_degree(v) == 0))
    levels = topological_level_counts(G, max_levels=6)
    for i, c in enumerate(levels):
        feats[f"level_count_{i}"] = float(c)
    feats["unlabelled_st_paths"] = float(count_st_paths_dag(G))
    return feats

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="infile", required=True)
    p.add_argument("--out", dest="outfile", required=True)
    args = p.parse_args()
