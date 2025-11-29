# flipclass-ai — placed under software/

This directory contains a CPU-only pipeline to:

- load your examples from `data/db` (Sage .sobj files; loader uses Sage.load),
- canonicalize directed graphs (remove labels) and group by isomorphism type,
- extract structural features for each isomorphism-type representative,
- train a CPU regressor to predict the number of increasing-labelled s→t paths,
- attempt to derive a closed-form symbolic formula via sparse polynomial regression.

Quick start (run with Sage where noted)
1. Install Python deps (conda/mamba/pip) — see software/requirements.txt.
2. Inspect dataset (must use Sage's Python):
   sage -python software/src/check_data.py --db-dir data/db --sample 5
3. Build canonical dataset:
   sage -python software/src/build_dataset.py --db-dir data/db --out dataset/canonical_dataset.pkl
4. Extract features:
   python software/src/features.py --in dataset/canonical_dataset.pkl --out dataset/features.pkl
5. Train model:
   python software/src/train.py --in dataset/features.pkl --out models/rf_model.joblib --n-jobs 32
6. Derive polynomial formula:
   python software/src/symbolic_regression.py --in dataset/features.pkl --out models/symbolic_formula.txt

Notes
- The loader is strict and uses Sage.load only; do not attempt to run the loader with plain Python.
- If you prefer, create a branch and open a PR so you can review changes before merging into main.
