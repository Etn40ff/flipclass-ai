#!/usr/bin/env python3
"""
Sparse online linear regressor using monomial triangular indexing.

This module provides:
 - triangular_index(i,j): deterministic mapping (i,j) -> index in N
 - SparseSGDRegressor: very small, dictionary-backed SGD for squared error

The index mapping follows your requested ordering:
  for d = 0..infty:
    for i = d..0 (descending):
      j = d - i
      yield (i,j)
Index formula: index = d*(d+1)//2 + (d - i)

The regressor stores weights in a dict[int->float] and an intercept float.
It supports partial_fit on sparse inputs represented as dict[int->float].
"""
from typing import Dict, Iterable, Tuple
import math
import joblib

def triangular_index(i: int, j: int) -> int:
    if i < 0 or j < 0:
        raise ValueError("exponents must be non-negative")
    d = i + j
    # offset for degrees < d: sum_{k=0..d-1} (k+1) = d*(d+1)/2
    offset = d * (d + 1) // 2
    pos_within = d - i  # 0..d (i descends from d..0)
    return offset + pos_within

def coef_map_to_indexed_sparse(coef_map: Dict[str, float]) -> Dict[int, float]:
    """
    Convert coef_map with keys 'i_j' -> coeff to sparse mapping index -> coeff (float).
    Malformed keys are ignored. Coefficients are cast to float.
    """
    out: Dict[int, float] = {}
    for key, val in coef_map.items():
        try:
            i_s, j_s = key.split("_")
            i = int(i_s); j = int(j_s)
            idx = triangular_index(i, j)
            out[idx] = out.get(idx, 0.0) + float(val)
        except Exception:
            # ignore malformed entries
            continue
    return out

class SparseSGDRegressor:
    """
    Simple online SGD regressor with L2 regularization (Ridge-like) that stores
    weights in a dict keyed by integer feature index.

    Loss: 0.5 * (y - w.x - b)^2  + 0.5 * l2 * ||w||^2

    Update rule (per-sample, learning rate eta):
      for each active feature f with value x_f:
        w_f <- w_f + eta * (error * x_f - l2 * w_f)
      b <- b + eta * error

    This is plain SGD for squared error with L2 shrinkage applied in the update.
    """
    def __init__(self, eta0: float = 1e-3, l2: float = 1e-5, normalize: bool = True):
        self.weights: Dict[int, float] = {}
        self.intercept: float = 0.0
        self.eta0 = float(eta0)
        self.l2 = float(l2)
        self.iterations = 0
        self.normalize = bool(normalize)

    def predict_sparse(self, x_sparse: Dict[int, float]) -> float:
        s = self.intercept
        for k, v in x_sparse.items():
            w = self.weights.get(k)
            if w is not None:
                s += w * v
        return s

    def partial_fit_one(self, x_sparse: Dict[int, float], y: float):
        """
        Single-sample update.
        x_sparse: dict[index->value]
        y: scalar target
        """
        # optional per-sample L2 normalization
        if self.normalize:
            norm = math.sqrt(sum(float(v) * float(v) for v in x_sparse.values()))
            if norm > 0:
                x_scaled = {k: float(v) / norm for k, v in x_sparse.items()}
            else:
                x_scaled = x_sparse
        else:
            x_scaled = {k: float(v) for k, v in x_sparse.items()}

        pred = self.predict_sparse(x_scaled)
        error = float(y) - pred

        # learning rate schedule: simple inverse sqrt
        self.iterations += 1
        eta = self.eta0 / math.sqrt(max(1, self.iterations))

        # update weights for active keys only
        for k, xv in x_scaled.items():
            w = self.weights.get(k, 0.0)
            # gradient = - error * xv + l2 * w (for squared loss w.r.t. w)
            # SGD step (minimize): w <- w - eta * grad = w + eta * (error * xv - l2*w)
            w_new = w + eta * (error * xv - self.l2 * w)
            # tiny-threshold pruning to keep dict small
            if abs(w_new) < 1e-12:
                self.weights.pop(k, None)
            else:
                self.weights[k] = w_new

        # update intercept (bias)
        self.intercept += eta * error

    def partial_fit_batch(self, X_sparse: Iterable[Dict[int, float]], y_iterable: Iterable[float], verbose: bool = True):
        count = 0
        for x, y in zip(X_sparse, y_iterable):
            self.partial_fit_one(x, y)
            count += 1
            if verbose and count % 1000 == 0:
                print(f"trained on {count} samples, weights={{len(self.weights)}}")
        return count

    def score_on_batch(self, X_sparse: Iterable[Dict[int, float]], y_iterable: Iterable[float]) -> Tuple[float, float]:
        """
        Return (mse, n) on the provided batch (for monitoring).
        """
        sse = 0.0
        n = 0
        for x, y in zip(X_sparse, y_iterable):
            p = self.predict_sparse(x)
            err = float(y) - p
            sse += err * err
            n += 1
        return (sse / n if n else float('nan'), n)

    def to_dict(self) -> Dict:
        return {
            "weights": self.weights,
            "intercept": self.intercept,
            "eta0": self.eta0,
            "l2": self.l2,
            "iterations": self.iterations,
            "normalize": self.normalize,
            "indexing": "triangular_index(i,j) with order by total degree then descending i",
        }

    @classmethod
    def from_dict(cls, d: Dict):
        m = cls(eta0=d.get("eta0", 1e-3), l2=d.get("l2", 1e-5), normalize=d.get("normalize", True))
        m.weights = dict(d.get("weights", {}))
        m.intercept = float(d.get("intercept", 0.0))
        m.iterations = int(d.get("iterations", 0))
        return m

    def save(self, path: str):
        joblib.dump({"model": self.to_dict()}, path)

    @classmethod
    def load(cls, path: str):
        raw = joblib.load(path)
        d = raw.get("model") if isinstance(raw, dict) and "model" in raw else raw
        return cls.from_dict(d)

--- software/src/train_streaming.py ---
#!/usr/bin/env python3
"""
Streaming / incremental trainer using sparse triangular-index representation.

This trainer reads the raw polynomial examples pickle (polys.pkl) produced by
extract_features_parallel.py (each example contains 'poly_coefs' mapping
'i_j' -> coeff and 'target') and trains a SparseSGDRegressor that can grow
to accommodate arbitrarily large monomials (no fixed max-degree).

Example:
  python software/src/train_streaming.py --from-polys dataset/polys.pkl --model models/sparse_model.joblib --eta0 1e-3 --l2 1e-5 --batch-size 2000

Notes:
 - The model stores weights as a dict mapping triangular-index -> coefficient.
 - Optionally normalize each sample by its L2 norm (default True).
"""
import argparse
import pickle
import joblib
from typing import Dict
from software.src.online_model import coef_map_to_indexed_sparse, SparseSGDRegressor
import math

def stream_train(polys_pickle: str, model_path: str, batch_size: int = 2000, eta0: float = 1e-3, l2: float = 1e-5, normalize: bool = True):
    with open(polys_pickle, "rb") as f:
        examples = pickle.load(f)
    if not isinstance(examples, list):
        raise SystemExit("expected list of examples in polys pickle")

    model = SparseSGDRegressor(eta0=eta0, l2=l2, normalize=normalize)

    N = len(examples)
    start = 0
    while start < N:
        end = min(N, start + batch_size)
        batch = examples[start:end]
        X_sparse = []
        y = []
        for ex in batch:
            coefs = ex.get("poly_coefs", {}) or {}
            xsp = coef_map_to_indexed_sparse(coefs)
            X_sparse.append(xsp)
            y.append(float(ex.get("target", 0)))
        model.partial_fit_batch(X_sparse, y, verbose=False)
        mse, cnt = model.score_on_batch(X_sparse, y)
        print(f"Trained on {start}:{end} (batch {cnt}). batch_mse={{mse:.6g}}, total_weights={{len(model.weights)}}")
        start = end

    # Save model
    meta = {
        "model": model.to_dict(),
        "format_version": 1,
    }
    joblib.dump(meta, model_path)
    print("Saved sparse model to", model_path)
    print("Final weights count:", len(model.weights))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from-polys", dest="polys", required=True, help="pickle produced by extract_features_parallel.py")
    p.add_argument("--model", required=True, help="output joblib model path")
    p.add_argument("--batch-size", type=int, default=2000)
    p.add_argument("--eta0", type=float, default=1e-3)
    p.add_argument("--l2", type=float, default=1e-5)
    p.add_argument("--no-normalize", dest="normalize", action="store_false", help="disable per-sample L2 normalization")
    args = p.parse_args()

    stream_train(args.polys, args.model, batch_size=args.batch_size, eta0=args.eta0, l2=args.l2, normalize=args.normalize)

if __name__ == "__main__":
    main()

--- software/src/predict.py ---
#!/usr/bin/env python3
"""
Predict with a sparse triangular-index model on arbitrary-degree polynomials.

Usage:
  # single polynomial (string parsed via build_dataset.poly_to_coefs)
  python software/src/predict.py --model models/sparse_model.joblib --poly "3*x^2*y + 2*x*y^2 - 5"

  # predict from a polys.pkl (each example has 'poly_coefs') and write CSV
  python software/src/predict.py --model models/sparse_model.joblib --from-polys dataset/polys.pkl --out preds.csv
"""
import argparse
import joblib
import pickle
import csv
from software.src.build_dataset import poly_to_coefs
from software.src.online_model import coef_map_to_indexed_sparse, SparseSGDRegressor

def load_model(path: str) -> SparseSGDRegressor:
    raw = joblib.load(path)
    d = raw.get("model") if isinstance(raw, dict) and "model" in raw else raw
    return SparseSGDRegressor.from_dict(d)

def predict_one(model: SparseSGDRegressor, poly_str: str) -> float:
    coefs = poly_to_coefs(poly_str)
    xsp = coef_map_to_indexed_sparse(coefs)
    return model.predict_sparse(xsp)

def predict_from_polys(model: SparseSGDRegressor, polys_pickle: str, out_csv: str = None):
    with open(polys_pickle, "rb") as f:
        examples = pickle.load(f)
    results = []
    for ex in examples:
        coefs = ex.get("poly_coefs", {}) or {}
        xsp = coef_map_to_indexed_sparse(coefs)
        pred = model.predict_sparse(xsp)
        results.append({"source_file": ex.get("source_file"), "target": int(ex.get("target", 0)), "pred": float(pred)})
    if out_csv:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["source_file", "target", "pred"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--poly", default=None)
    p.add_argument("--from-polys", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    model = load_model(args.model)

    if args.poly:
        pred = predict_one(model, args.poly)
        print("pred_raw:", pred, "pred_round:", int(round(pred)))
    elif args.from_polys:
        res = predict_from_polys(model, args.from_polys, out_csv=args.out)
        print("Wrote predictions:", len(res), "to", args.out if args.out else "stdout")
    else:
        print("Specify --poly or --from-polys")

if __name__ == "__main__":
    main()