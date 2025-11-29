#!/usr/bin/env python3
"""
Build a canonical dataset from files created with Sage's save().

This loader uses Sage's load() exclusively (no decoding heuristics).
Run with Sage's Python (sage -python ...) or ensure sage.all is importable.

Output: pickle file containing a list of dicts:
  {'canonical_id': int, 'graph': networkx.DiGraph, 'members': [...], 'count': int}
"""
import os
import argparse
import pickle
from typing import Any, List, Tuple
import networkx as nx
from sage.all import load  # requires Sage
from software.src.canonicalize import group_isomorphism_classes

def load_file(path: str) -> Any:
    # Strict: use Sage's load only
    return load(path)

def to_networkx_graph(obj: Any) -> nx.DiGraph:
    """
    Convert the loaded graph object to a networkx.DiGraph.
    Accepts:
      - networkx.DiGraph (returns a copy)
      - Python dict in two forms:
          {'nodes': [...], 'edges': [(u,v), ...]}
          adjacency dict {u: [v1, v2, ...], ...}
      - Common Sage graph-like objects: try .to_networkx(), .to_networkx_graph(),
        or use vertices()/edges() methods.
    Raises RuntimeError on unsupported types.
    """
    if isinstance(obj, nx.DiGraph):
        return obj.copy()
    # Python dict encodings
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
    # Sage graph objects: try common conversion methods
    if hasattr(obj, "to_networkx"):
        try:
            nxg = obj.to_networkx()
            if not isinstance(nxg, nx.DiGraph):
                nxg = nx.DiGraph(nxg)
            return nxg
        except Exception:
            pass
    if hasattr(obj, "to_networkx_graph"):
        try:
            nxg = obj.to_networkx_graph()
            if not isinstance(nxg, nx.DiGraph):
                nxg = nx.DiGraph(nxg)
            return nxg
        except Exception:
            pass
    if hasattr(obj, "vertices") and hasattr(obj, "edges"):
        try:
            nodes = list(obj.vertices())
            raw_edges = list(obj.edges())
            edges = []
            for e in raw_edges:
                try:
                    if len(e) >= 2:
                        edges.append((e[0], e[1]))
                    else:
                        pass
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
        except Exception:
            pass
    raise RuntimeError(f"Unsupported graph object of type {type(obj)}; cannot convert to networkx.DiGraph")

def extract_examples_from_file(path: str) -> List[Tuple[nx.DiGraph, int]]:
    data = load_file(path)
    if not isinstance(data, dict):
        raise RuntimeError(f"file {path} does not contain a top-level dict (got {type(data)})")
    examples: List[Tuple[nx.DiGraph, int]] = []
    for k, v in data.items():
        try:
            target = len(k)
        except Exception:
            raise RuntimeError(f"Could not compute len(key) for key: {repr(k)[:200]}")
        G = to_networkx_graph(v)
        # drop attributes to focus on unlabelled structure
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from(G.edges())
        examples.append((H, int(target)))
    return examples

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db-dir", default="data/db")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    if not os.path.isdir(args.db_dir):
        raise SystemExit("db-dir not found: " + args.db_dir)
    files = sorted(os.listdir(args.db_dir))
    all_graphs = []
    all_targets = []
    for fn in files:
        fp = os.path.join(args.db_dir, fn)
        print("loading", fp)
        exs = extract_examples_from_file(fp)
        for G, t in exs:
            all_graphs.append(G)
            all_targets.append(int(t))

    mapping, reps = group_isomorphism_classes(all_graphs)

    canonical_list = []
    for cid, members in mapping.items():
        counts = [all_targets[i] for i in members]
        uniq = sorted(set(counts))
        if len(uniq) > 1:
            print(f"WARNING: isomorphism class {cid} has multiple counts {uniq}; storing first")
        rep_graph = all_graphs[members[0]]
        canonical_list.append({
            "canonical_id": int(cid),
            "graph": rep_graph,
            "members": members,
            "count": int(uniq[0])
        })
    with open(args.out, "wb") as f:
        pickle.dump(canonical_list, f)
    print("wrote", len(canonical_list), "canonical examples to", args.out)

if __name__ == "__main__":
    main()
