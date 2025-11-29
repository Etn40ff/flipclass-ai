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
    # offset for degrees < d: sum_{k=0..d-1} (k+1) = d*(d+1)//2
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
                print(f"trained on {count} samples, weights={len(self.weights)}")
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
