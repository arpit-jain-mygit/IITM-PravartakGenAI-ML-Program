#!/usr/bin/env bash
# Reproduce the whole capstone with one command.
#   ./run_all.sh                    live run (OPENAI_API_KEY in .env; OPENAI_BASE_URL for Vocareum)
#   RETAIL_OFFLINE=1 ./run_all.sh   no-key dry run (local embeddings, deterministic generator; writes outputs_offline/)
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-$PWD/../.venv/bin/python}"; [ -x "$PY" ] || PY=python3
NAME="Capstone_Graded Project_Jain"

if [ "${RETAIL_OFFLINE:-0}" != "1" ]; then
  if ! grep -q "^OPENAI_API_KEY=." .env 2>/dev/null || grep -q "^OPENAI_API_KEY=paste-" .env; then
    echo "Put your OPENAI_API_KEY in capstone_project/.env first (or run with RETAIL_OFFLINE=1)"; exit 1
  fi
fi

# 1. regression tests (always offline, no key needed)
$PY -m pytest -q --color=no -p no:warnings tests

# 2. execute the notebooks in order (01 -> 06); live outputs are saved inside the notebooks
cd notebooks
for nb in 01_data_preparation 02_classical_ml 03_rag_recommendation 04_policy_qa 05_agentic_assistant 06_evaluation; do
  echo "== running $nb"
  if [ "${RETAIL_OFFLINE:-0}" = "1" ]; then
    mkdir -p ../outputs_offline/notebooks
    $PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=1800 --output "../outputs_offline/notebooks/$nb.ipynb" "$nb.ipynb"
  else
    $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 "$nb.ipynb"
  fi
done
cd ..

# 3. short report (PDF) from the saved outputs
[ -f make_report.py ] && $PY make_report.py      # report builder lives in the repo only, not in the zip

# 4. submission zip (live run only): everything except keys, caches and offline artefacts
if [ "${RETAIL_OFFLINE:-0}" != "1" ]; then
  STAGE=$(mktemp -d); mkdir "$STAGE/$NAME"
  rsync -a --exclude ".env" --exclude ".cache*" --exclude "outputs_offline" --exclude "__pycache__" \
        --exclude ".pytest_cache" --exclude "make_report.py" ./ "$STAGE/$NAME/capstone_project/"
  cp outputs/Capstone_Report.pdf "$STAGE/$NAME/"
  rm -f "../$NAME.zip"; (cd "$STAGE" && zip -qr - "$NAME") > "../$NAME.zip"
  echo "Ready: ../$NAME.zip  (report: outputs/Capstone_Report.pdf)"
fi
