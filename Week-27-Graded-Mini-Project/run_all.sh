#!/usr/bin/env bash
# Builds every deliverable with one command.
#   ./run_all.sh                    live run with OpenAI (OPENAI_API_KEY in agentic-policy-assistant/.env)
#   POLICY_OFFLINE=1 ./run_all.sh   dry run with the offline stand-in model (no key, writes *_offline dirs only)
set -euo pipefail
cd "$(dirname "$0")"
PY="$PWD/.venv/bin/python"
NAME="Week 27_Graded Mini Project_Jain"
cd agentic-policy-assistant

# 1. unit + acceptance + budget tests (always offline, no key needed)
$PY -m pytest -q --color=no tests -p no:warnings

# 2. the notebook drives everything: tools, agent, eval v1, cold/warm runs, recovery, graph, critic, PNGs
if [ "${POLICY_OFFLINE:-0}" = "1" ]; then
  mkdir -p runs_offline
  $PY -m jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=3600 --output "runs_offline/notebook_offline.ipynb" "$NAME.ipynb"
else
  if ! grep -q "^OPENAI_API_KEY=." .env || grep -q "^OPENAI_API_KEY=paste-" .env; then
    echo "Put your OPENAI_API_KEY in agentic-policy-assistant/.env first"; exit 1
  fi
  rm -rf runs results          # fresh, reproducible live run (cold memory, empty cache)
  $PY -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 "$NAME.ipynb"
fi
cd ..

# 3. fill the course template -> docx + pdf, and the 2-page analysis summary
$PY make_report.py
$PY make_summary.py

# 4. submission zip: PDF + notebook + results PNGs + the whole repo (without keys, caches or venv)
if [ "${POLICY_OFFLINE:-0}" != "1" ]; then
  STAGE=$(mktemp -d); mkdir "$STAGE/$NAME"
  cp "$NAME.pdf" "${NAME}_Summary.pdf" "agentic-policy-assistant/$NAME.ipynb" "$STAGE/$NAME/"
  cp -R agentic-policy-assistant/results "$STAGE/$NAME/results"
  rsync -a --exclude ".env" --exclude "__pycache__" --exclude ".pytest_cache" --exclude "runs_offline" \
        --exclude "results_offline" --exclude "*.sqlite" agentic-policy-assistant "$STAGE/$NAME/"
  rm -f "$NAME.zip"; (cd "$STAGE" && zip -qr - "$NAME") > "$NAME.zip"
  echo "Ready: $NAME.pdf, ${NAME}_Summary.pdf, $NAME.zip, agentic-policy-assistant/$NAME.ipynb, agentic-policy-assistant/results/"
fi
