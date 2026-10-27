# Repository tooling

These scripts build the reports and the submission zips. They live in this GitHub repository only and are **not** included in any submission zip.

| Assignment | Script | What it does |
|---|---|---|
| Week 7 | `Week-07-Graded-Mini-Project/package.sh` | Rebuilds `report.pdf` (via `make_report.py`) and the submission zip |
| Week 9 | `Week-09-Graded-Mini-Project/package.sh` | Rebuilds the report PDF/DOCX (`make_report.py`, `report_helpers.py`) and the zip |
| Week 19 | `Week-19-Graded-Mini-Project/package.sh` | Rebuilds the report and the 2-page analysis PDF (`make_report.py`, `report_helpers.py`, `observations.json`) and the zip |
| Week 24 | `Week-24-Graded-Mini-Project/run_all.sh`, `check_setup.py` | Live run end to end; pre-flight check of the keys and the Gemini model |
| Week 27 | `Week-27-Graded-Mini-Project/run_all.sh` | Tests, notebook, results PNGs, full report (`make_report.py`), 2-page summary (`make_summary.py`) and zip |
| Capstone | `Capstone-Retail-Decision-Assistant/capstone_project/run_all.sh` | Tests, notebooks 01-06, report (`make_report.py`) and zip |

**Report dependencies:** the report scripts need `python-docx` and `reportlab`, plus LibreOffice for the DOCX → PDF conversion (`brew install --cask libreoffice`).

## Week 24 workflow (RAG with Gemini + OpenAI embeddings)

1. **Add the keys.** In `Week-24-Graded-Mini-Project/.env`, set `GOOGLE_API_KEY` and `OPENAI_API_KEY`. `GEMINI_MODEL=gemini-3.5-flash-lite` is the model used for the graded run.
2. **Check the setup:** `.venv/bin/python check_setup.py`. It makes 2 tiny calls, and if a listed model has been retired it prints the replacement to use.
3. **Run:** `./run_all.sh`. It takes about 15-20 minutes on the Gemini free tier. If Gemini returns 503/504 overload errors, re-run it; finished answers are cached and reused.
4. **Review the scores.**
   1. Open `logs/manual_scores.csv` next to `logs/run_log.csv`.
   2. Correct any draft score you disagree with.
   3. Set `reviewed = True` on all 70 rows.
   4. Run `./run_all.sh` again (cached, so no API calls).
5. **Test without keys:** `RAG_DRY_RUN=1 ./run_all.sh` writes only to `dryrun/`.

## Week 27 and Capstone (OpenAI)

1. **Add the key.** Put `OPENAI_API_KEY` in `agentic-policy-assistant/.env` (Week 27) or `capstone_project/.env` (Capstone). For Vocareum, also add `OPENAI_BASE_URL`.
2. **Run:** `./run_all.sh` in the assignment folder.
3. **Test without keys:** `POLICY_OFFLINE=1 ./run_all.sh` (Week 27) or `RETAIL_OFFLINE=1 ./run_all.sh` (Capstone).

## What each zip contains

Only the deliverables:
- the notebook(s) with outputs
- the report PDF(s)
- the results/outputs
- the code the notebooks import
- the data
- the README and requirements
- `.env.example`, which records the keys and model needed, with no real keys in it

**Never included:** `.env`, caches of keys, the report-building scripts, `run_all.sh`, `check_setup.py`, `package.sh`.

**Capstone zip:** it contains the course-provided data files, so it is kept out of this public repository and exists only locally for upload.
