#!/usr/bin/env python3
"""
Attempt to derive a closed-form formula from features -> count.

Methods:
- lasso: polynomial features + LassoCV (fast)
- gplearn: optional genetic-programming symbolic regression (slower)
"""

import argparse
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LassoCV
import sympy as sp
from sklearn.model_selection import train_test_split

def lasso_polynomial_formula(X, y, degree=2):
    pf = PolynomialFeatures(degree=degree, include_bias=True)
    Xp = pf.fit_transform(X)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xp)
    lasso = LassoCV(cv=5, n_jobs=-1, random_state=42, max_iter=20000)
    lasso.fit(Xs, y)
    coef = lasso.coef_
    intercept = lasso.intercept_
    syms = {col: sp.Symbol(col) for col in X.columns}
    expr = sp.Float(intercept)
    powers = pf.powers_
    feature_names = pf.get_feature_names_out(X.columns)
    for coeff, power_vec in zip(coef, powers):
        if abs(coeff) < 1e-12:
            continue
        term = sp.Float(coeff)
        for col, p in zip(X.columns, power_vec):
            if p == 0:
                continue
            term = term * (syms[col] ** int(p))
        expr = expr + term
    return sp.simplify(expr), lasso, pf, scaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    parser.add_argument("--method", choices=["lasso", "gplearn"], default="lasso")
    parser.add_argument("--degree", type=int, default=2)
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']

    if args.method == "lasso":
        expr, model, pf, scaler = lasso_polynomial_formula(X, y, degree=args.degree)
        s = "Derived polynomial (Lasso) formula:\n" + str(expr)
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write(s)
        print(s)
        print("Saved to", args.outfile)
    else:
        try:
            from gplearn.genetic import SymbolicRegressor
        except Exception:
            raise SystemExit("gplearn is not installed. Install with `pip install gplearn` to use method=gplearn")
        X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2, random_state=42)
        est = SymbolicRegressor(population_size=2000, generations=40, stopping_criteria=0.01,
                                p_crossover=0.7, p_subtree_mutation=0.1, p_hoist_mutation=0.05,
                                p_point_mutation=0.1, max_samples=0.9, verbose=1,
                                parsimony_coefficient=0.001, random_state=42, n_jobs=8)
        est.fit(X_train, y_train)
        prog = str(est._program)
        with open(args.outfile, "w", encoding="utf-8") as f:
            f.write("gplearn program:\n")
            f.write(prog)
        print("Best program:", prog)
        print("Saved to", args.outfile)

if __name__ == "__main__":
    main()