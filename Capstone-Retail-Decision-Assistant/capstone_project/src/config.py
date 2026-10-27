"""
Central configuration: paths, model names, seeds and the OFFLINE switch.

LIVE mode (default) uses an OpenAI-compatible endpoint (OpenAI or the Vocareum-provided key/base URL)
for embeddings and the LLM. OFFLINE mode (RETAIL_OFFLINE=1) swaps in a local open-source embedding model
and a deterministic extractive generator, so the whole project can be run and tested without any key.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

OFFLINE = os.getenv("RETAIL_OFFLINE") == "1"

# --- paths -----------------------------------------------------------------
RAW_CSV = ROOT / "data" / "raw" / "retail_interactions.csv"
POLICY_JSON = ROOT / "data" / "policies" / "retail_policies.json"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / ("outputs_offline" if OFFLINE else "outputs")
CACHE = ROOT / (".cache_offline" if OFFLINE else ".cache")
for p in (PROCESSED, OUTPUTS, CACHE):
    p.mkdir(parents=True, exist_ok=True)

# --- models ------------------------------------------------------------------
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
LOCAL_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # offline mode only
TEMPERATURE = 0.0
SEED = 42

# --- retrieval ---------------------------------------------------------------
TOP_K_PRODUCTS = 5
TOP_K_POLICIES = 3
# minimum cosine similarity for a hit to count as relevant (calibrated per embedding model in notebook 03)
MIN_PRODUCT_SIM = 0.30 if not OFFLINE else 0.25
# live value calibrated in notebook 04 (OpenAI embeddings): weakest correct policy match 0.345 (cracked watch
# glass -> POL_WARRANTY), strongest out-of-scope question 0.296 (store opening hours) -> midpoint 0.32
MIN_POLICY_SIM = 0.32 if not OFFLINE else 0.30

# --- agent -------------------------------------------------------------------
MAX_TOOL_CALLS = 3          # hard bound on tool calls per user query
