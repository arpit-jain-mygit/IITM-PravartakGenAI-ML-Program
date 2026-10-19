"""
Quick pre-flight check before running the experiments (makes 2 tiny API calls).

    python check_setup.py

- confirms GOOGLE_API_KEY and OPENAI_API_KEY are set in .env
- lists the Gemini models your key can use and checks that GEMINI_MODEL is one of them
- sends one short test prompt to Gemini and embeds one sentence with OpenAI
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()
ok = True

# --- 1. Keys present? ---
for k in ["GOOGLE_API_KEY", "OPENAI_API_KEY"]:
    v = os.getenv(k, "")
    good = bool(v) and not v.startswith("paste-")
    print(f"{k:15s}: {'found' if good else 'MISSING - add it to .env'}")
    ok &= good
if not ok:
    sys.exit(1)

import rag  # noqa: E402  (imported after .env is loaded)

# --- 2. Which Gemini models can this key use? ---
from google import genai  # installed with langchain-google-genai
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
names = sorted(m.name.replace("models/", "") for m in client.models.list()
               if "generateContent" in (m.supported_actions or []) and "flash" in m.name)
print("\nGemini 'flash' models available to your key:", ", ".join(names) or "none found")
if rag.GEMINI_MODEL not in names:
    print(f"!! GEMINI_MODEL '{rag.GEMINI_MODEL}' is not in that list. Add a line like "
          f"GEMINI_MODEL={names[0] if names else '<model>'} to .env and run this check again.")
    sys.exit(1)

# --- 3. One tiny call to each API ---
reply = rag.get_llm().invoke("Reply with the single word: ready")
print(f"Gemini ({rag.GEMINI_MODEL}) says: {reply.text.strip()!r}")
vec = rag.get_embeddings("openai").embed_query("setup check")
print(f"OpenAI {rag.OPENAI_EMBED_MODEL}: vector of {len(vec)} dimensions")
print("\nAll good - run ./run_all.sh")
