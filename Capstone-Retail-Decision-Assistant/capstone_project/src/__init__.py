"""Intelligent Retail Decision Assistant - source package.

macOS note: faiss-cpu and PyTorch (used by the offline embedding model) each bundle an OpenMP runtime.
Allowing the duplicate avoids a hard crash on macOS; it has no effect on Linux / Google Colab."""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
