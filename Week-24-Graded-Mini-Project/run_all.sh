#!/usr/bin/env bash
# One command for the whole assignment:
#   ./run_all.sh               live run  (needs GOOGLE_API_KEY + OPENAI_API_KEY in .env)
#   RAG_DRY_RUN=1 ./run_all.sh  test run with no API calls (writes only to dryrun/)
set -euo pipefail
cd "$(dirname "$0")"
PY=.venv/bin/python
NB="Week 24_Graded Mini Project_Jain.ipynb"
NAME="Week 24_Graded Mini Project_Jain"

# 1. corpus PDFs (deterministic)
$PY make_corpus.py

# 2. execute the notebook: experiments, logging, draft scores, summary
if [ "${RAG_DRY_RUN:-0}" = "1" ]; then
  mkdir -p dryrun
  $PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output "dryrun/notebook_dryrun.ipynb" "$NB"
else
  $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$NB"
fi

# 3. build the report -> .docx + .pdf
$PY make_report.py

# 4. submission zip (live run only)
if [ "${RAG_DRY_RUN:-0}" != "1" ]; then
  STAGE=$(mktemp -d); mkdir "$STAGE/$NAME"
  cp -R "$NAME.pdf" "$NB" README.md requirements.txt .env.example run_all.sh check_setup.py \
        rag.py make_corpus.py scoring.py summarize.py make_report.py report_helpers.py questions.json \
        corpus logs outputs cache "$STAGE/$NAME/"
  rm -f "$NAME.zip"; (cd "$STAGE" && zip -qr - "$NAME") > "$NAME.zip"
  echo "Ready: $NAME.pdf and $NAME.zip"
fi
