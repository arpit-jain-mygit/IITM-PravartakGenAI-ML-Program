"""Regression tests always run OFFLINE (local embeddings, deterministic generator): no key, no cost."""
import os
import sys
from pathlib import Path

os.environ["RETAIL_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src  # noqa: E402,F401  (sets macOS OpenMP flags)
