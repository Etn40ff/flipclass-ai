#!/usr/bin/env python3
"""
Train a CPU-only regression model on the extracted features.

Usage:
  python software/src/train.py --in dataset/features.pkl --out models/rf_model.joblib --n-jobs 32
"""

import argparse
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os
import multiprocessing

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="infile", required=True)
    parser.add_argument("--out", dest="outfile", required=True)
    parser.add_argument("--n-jobs", type=int, default=min(8, multiprocessing.cpu_count()))
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    with open(args.infile, "rb") as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    print("X shape:", X.shape, "y shape:", y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)
    model = RandomForestRegressor(n_estimators=200, n_jobs=args.n_jobs, random_state=42)
    print("Fitting RandomForest with n_jobs =", args.n_jobs)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("MAE:", mae, "MSE:", mse, "R2:", r2)
    os.makedirs(os.path.dirname(args.outfile) or ".", exist_ok=True)
    joblib.dump({'model': model, 'feature_columns': list(X.columns)}, args.outfile)
    base = os.path.splitext(args.outfile)[0]
    pd.DataFrame({'y_true': y_test, 'y_pred': y_pred}).to_csv(base + "_preds.csv", index=False)
    print("Saved model to", args.outfile)
    print("Predictions saved to", base + "_preds.csv")

if __name__ == "__main__":
    main()