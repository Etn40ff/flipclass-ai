# flipclass-ai — placed under software/

This directory contains a CPU-only pipeline to:

- load your examples from `data/db` (supports pickle, JSON, and Python-dict text),
- canonicalize directed graphs (remove labels) and group by isomorphism type,
- extract structural features for each isomorphism-type representative,
- train a CPU regressor to predict the number of increasing-labelled s→t paths,
- attempt to derive a closed-form symbolic formula via sparse polynomial regression
  or optional genetic programming.

Quick start (example, run on cluster or laptop)
1. Create a virtualenv and install dependencies:
   pip install -r software/requirements.txt

2. Inspect dataset format:
   python software/src/check_data.py --db-dir data/db --sample 5

3. Build canonical dataset:
   python software/src/build_dataset.py --db-dir data/db --out dataset/canonical_dataset.pkl --n-jobs 32

4. Extract features:
   python software/src/features.py --in dataset/canonical_dataset.pkl --out dataset/features.pkl

5. Train model (example):
   python software/src/train.py --in dataset/features.pkl --out models/rf_model.joblib --n-jobs 32

6. Try symbolic regression (Lasso polynomial):
   python software/src/symbolic_regression.py --in dataset/features.pkl --out models/symbolic_formula.txt --method lasso --degree 2

Notes
- All scripts are CPU-only and accept an --n-jobs argument to parallelize where relevant.
- If build_dataset.py warns about multiple different counts for the same isomorphism type, inspect those cases — they indicate the count depends on edge labels (or inconsistent data).