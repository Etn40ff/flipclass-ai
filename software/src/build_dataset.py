#!/usr/bin/env python3
"""
Utilities to canonicalize directed graphs to isomorphism classes (labels removed).

Main function: group_isomorphism_classes(graphs, n_jobs=1) -> dict:
  canonical_idx -> list of indices into graphs
  reps -> list of representative graphs

Strategy:
- compute cheap invariant signature (n_nodes, n_edges, deg hist, spectrum sample)
- bucket graphs by signature
- within each bucket, perform exact isomorphism checks using networkx.DiGraphMatcher
"""

import networkx as nx
import numpy as np
from collections import defaultdict

def invariant_signature(G, eigen_k=6):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    indeg = sorted([d for _, d in G.in_degree()])
    outdeg = sorted([d for _, d in G.out_degree()])
    try:
        A = nx.to_numpy_array(G, dtype=float)
        symA = (A + A.T) / 2.0
        vals = np.linalg.eigvals(symA)
        vals = np.sort(np.real(vals))
        if len(vals) > eigen_k:
            pick = list(vals[:eigen_k//2]) + list(vals[-(eigen_k - eigen_k//2):])
            vals_sig = tuple(np.round(np.real(pick), 6))
        else:
            vals_sig = tuple(np.round(vals, 6))
    except Exception:
        vals_sig = ()
    return (n, m, tuple(indeg), tuple(outdeg), vals_sig)

def are_isomorphic(G1, G2):
    gm = nx.algorithms.isomorphism.DiGraphMatcher(G1, G2)
    return gm.is_isomorphic()

def group_isomorphism_classes(graphs, n_jobs=1):
    """
    graphs: list of networkx.DiGraph objects
    returns: mapping: cid -> list(indices), reps: list of representative graphs
    """
    sigs = [invariant_signature(G) for G in graphs]
    buckets = defaultdict(list)
    for idx, sig in enumerate(sigs):
        buckets[sig].append(idx)

    classes = []
    reps = []

    for sig, indices in buckets.items():
        reps_in_bucket = []
        classes_in_bucket = []
        for i in indices:
            G = graphs[i]
            placed = False
            for j, rep_idx in enumerate(reps_in_bucket):
                if are_isomorphic(G, graphs[rep_idx]):
                    classes_in_bucket[j].append(i)
                    placed = True
                    break
            if not placed:
                reps_in_bucket.append(i)
                classes_in_bucket.append([i])
        for c in classes_in_bucket:
            reps.append(graphs[c[0]])
            classes.append(c)

    mapping = {cid: members for cid, members in enumerate(classes)}
    return mapping, reps
"""

File: software/src/build_dataset.py
"""
"""# /usr/bin/env python3
"""
Build canonical dataset grouping graphs by isomorphism type.

Reads all files under data/db (supports pickle, json, and python dict text). For each top-level key/value pair:
  - target = cardinality of the key (the set of increasing paths)
  - load the value into a networkx.DiGraph (attempt several heuristics)
Then group graphs by isomorphism type and produce a canonical_dataset list:
  list of dicts: {'canonical_id': int, 'graph': representative_graph, 'members': [indices], 'count': int}
"""

import os
import argparse
import pickle
import json
import ast
import networkx as nx
from collections import defaultdict
from software.src.canonicalize import group_isomorphism_classes

def try_load(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in (".pkl", ".pickle"):
        with open(path, "rb") as f:
            return pickle.load(f)
    elif ext in (".json",):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            return ast.literal_eval(text)
        except Exception as e:
            raise RuntimeError(f"Unsupported file format for {path}: {e}")

def to_networkx_graph(obj):
    """
    Heuristics:
    - If obj is a networkx.DiGraph -> use directly
    - If obj is adjacency dict: {u: [v1, v2, ...], ...}
    - If obj is list of edges: [(u,v), ...]
    - If obj is dict with 'nodes' and 'edges' keys
    """
    if isinstance(obj, nx.DiGraph):
        return obj
    G = nx.DiGraph()
    if isinstance(obj, dict):
        is_adj = all(isinstance(v, (list, tuple, set)) for v in obj.values())
        if is_adj:
            for u, vs in obj.items():
                G.add_node(u)
                for v in vs:
                    G.add_edge(u, v)
            return G
        if 'nodes' in obj and 'edges' in obj:
            G.add_nodes_from(obj['nodes'])
            G.add_edges_from(obj['edges'])
            return G
    if isinstance(obj, (list, tuple)):
        if all(isinstance(e, (list, tuple)) and len(e) >= 2 for e in obj):
            G.add_edges_from([tuple(e[:2]) for e in obj])
            return G
    raise RuntimeError("Could not convert object to DiGraph; inspect with software/src/check_data.py")

def extract_examples_from_file(path):
    data = try_load(path)
    examples = []
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                target = len(k)
            except Exception:
                try:
                    key_obj = ast.literal_eval(repr(k))
                    target = len(key_obj)
                except Exception:
                    raise RuntimeError(f"Could not compute cardinality of key: {k}")
            G = to_networkx_graph(v)
            examples.append((G, int(target)))
    else:
        raise RuntimeError(f"Top-level file content not a dict in {path}")
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-dir", default="data/db")
    parser.add_argument("--out", required=True, help="output pickle for canonical dataset")
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()

    if not os.path.isdir(args.db_dir):
        raise SystemExit("db-dir not found: " + args.db_dir)
    files = sorted(os.listdir(args.db_dir))
    all_graphs = []
    all_targets = []
    for fn in files:
        fp = os.path.join(args.db_dir, fn)
        print("loading", fp)
        try:
            exs = extract_examples_from_file(fp)
        except Exception as e:
            print("ERROR loading", fp, ":", e)
            continue
        for G, t in exs:
            H = nx.DiGraph()
            H.add_nodes_from(G.nodes())
            H.add_edges_from(G.edges())
            all_graphs.append(H)
            all_targets.append(int(t))
    print("total examples:", len(all_graphs))

    mapping, reps = group_isomorphism_classes(all_graphs, n_jobs=args.n_jobs)
    print("num iso classes:", len(mapping))

    canonical_list = []
    for cid, members in mapping.items():
        counts = [all_targets[i] for i in members]
        uniq_counts = sorted(set(counts))
        if len(uniq_counts) > 1:
            print(f"WARNING: class {cid} has members with different counts: {uniq_counts}")
        rep_graph = all_graphs[members[0]]
        canonical_list.append({
            'canonical_id': int(cid),
            'graph': rep_graph,
            'members': members,
            'count': int(uniq_counts[0])
        })

    with open(args.out, "wb") as f:
        pickle.dump(canonical_list, f)
    print("wrote canonical dataset to", args.out)

if __name__ == "__main__":
    main()