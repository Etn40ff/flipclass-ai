#!/usr/bin/env python3
"""
Convenience alias for older naming: calls extract_features_parallel.py.

Usage:
  sage -python software/src/features_parallel.py --db-dir data/db --out dataset/polys.pkl --n-jobs 8
"""
import sys
from pathlib import Path

# delegate to extract_features_parallel
script = Path(__file__).with_name("extract_features_parallel.py")
if not script.exists():
    print("extract_features_parallel.py not found in same directory")
    sys.exit(2)

# re-exec python with the target script and same args
args = [sys.executable, str(script)] + sys.argv[1:]
import os
os.execv(sys.executable, args)