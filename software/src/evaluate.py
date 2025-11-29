#!/usr/bin/env python3
"""
Evaluate a saved model on a feature dataset.

Usage:
  python software/src/evaluate.py --model models/rf_model.joblib --features dataset/features.pkl
"""

import argparse
import pickle
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--features", required=True)
    args = parser.parse_args()

    mdl = joblib.load(args.model)
    with open(args.features, "rb") as f:
        data = pickle.load(f)
    X = data['X']
    y = data['y']
    cols = mdl.get('feature_columns', None) or X.columns
    model = mdl['model']
    y_pred = model.predict(X[cols])
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    print("MAE:", mae, "MSE:", mse, "R2:", r2)

if __name__ == "__main__":
    main()