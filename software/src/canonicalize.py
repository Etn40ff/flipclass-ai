#!/usr/bin/env python3
"""
Canonicalization utilities.

Strategy:
- Bucket graphs by inexpensive signature (n_nodes, n_edges, sorted in-degree seq, sorted out-degree seq).
- Within each bucket perform exact isomorphism checks using networkx.DiGraphMatcher.
"""
from typing import List, Tuple, Dict
import networkx as nx
from collections import defaultdict

Signature = Tuple[int, int, Tuple[int, ...], Tuple[int, ...]]

def signature(G: nx.DiGraph) -> Signature:
    n = G.number_of_nodes()
    m = G.number_of_edges()
    indeg = tuple(sorted(d for _, d in G.in_degree()))
    outdeg = tuple(sorted(d for _, d in G.out_degree()))
    return (n, m, indeg, outdeg)

def group_isomorphism_classes(graphs: List[nx.DiGraph]) -> Tuple[Dict[int, List[int]], List[nx.DiGraph]]:
    """
    Return mapping cid -> list of indices into graphs, and a list of representative graphs.
    """
    buckets = defaultdict(list)
    sigs = [signature(G) for G in graphs]
    for i, sig in enumerate(sigs):
        buckets[sig].append(i)

    classes = []
    reps = []
    for sig, idxs in buckets.items():
        reps_in_bucket = []
        classes_in_bucket = []
        for i in idxs:
            G = graphs[i]
            placed = False
            for j, rep_idx in enumerate(reps_in_bucket):
                H = graphs[rep_idx]
                gm = nx.algorithms.isomorphism.DiGraphMatcher(G, H)
                if gm.is_isomorphic():
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
