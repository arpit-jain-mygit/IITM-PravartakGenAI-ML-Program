#!/usr/bin/env bash
# Repo tooling (kept in GitHub, NOT packed into the submission zip):
# rebuilds the report PDF from the notebook outputs, then builds the submission zip.
#   ./package.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-../.venv/bin/python}"
NAME="Week 7_Graded Mini Project_Jain"

# 1. report PDF from outputs/ (make_report.py is repo-only)
$PY make_report.py

# 2. submission zip: deliverables only - no package.sh, make_report.py, report_helpers.py or observations.json
STAGE=$(mktemp -d); mkdir "$STAGE/$NAME"
cp -R notebook.ipynb report.pdf requirements.txt model.pkl README.md model_metadata.json attrition_features.py predict.py WA_Fn-UseC_-HR-Employee-Attrition.csv outputs "$STAGE/$NAME/"
rm -f "$NAME.zip"; (cd "$STAGE" && zip -qr - "$NAME") > "$NAME.zip"
echo "Ready: $NAME.zip ($(unzip -Z1 "$NAME.zip" | grep -vc '/$') files)"
