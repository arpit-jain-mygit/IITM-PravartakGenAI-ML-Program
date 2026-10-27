# Week 27 Graded Mini Project: Policy Assistant Agent

IITM Pravartak - Professional Certificate Programme in Generative AI and Machine Learning

The code repository is [agentic-policy-assistant/](agentic-policy-assistant/). Its README covers the design, the CLI and the limitations.

## Steps to run (once you have an OpenAI key)

**1. Add the key.** In `agentic-policy-assistant/.env`, replace the placeholder:

```text
OPENAI_API_KEY=sk-...
```

For Vocareum, also add `OPENAI_BASE_URL=...`.

**2. Build everything.** This takes about 5 minutes and makes about 60 short gpt-4o-mini calls, costing a few cents:

```bash
cd Week-27-Graded-Mini-Project && ./run_all.sh
```

It runs, in order:
1. the 22 pytest tests
2. the notebook: tools, agent trace, Evaluation V1, error recovery, cold run, warm run, graph budget and retry, critic, SQLite checkpoints and the evaluation table
3. the results PNGs
4. the filled template as DOCX and PDF
5. the submission zip

**3. Check the outputs:**
- `Week 27_Graded Mini Project_Jain.pdf`: the template, filled with your live results
- `Week 27_Graded Mini Project_Jain_Summary.pdf`: the 2-page analysis summary (results table, cold vs warm, recovery, memory, limitations)
- `agentic-policy-assistant/Week 27_Graded Mini Project_Jain.ipynb`: the notebook with all outputs
- `agentic-policy-assistant/results/*.png`: the screenshots and charts
- `Week 27_Graded Mini Project_Jain.zip`: all of the above plus the code repo, without the key

**4. Upload the PDF, the .ipynb and the zip.** Only 1 attempt is allowed.

The two report scripts (`make_report.py`, `make_summary.py`) also need `pip install -r report-requirements.txt` (python-docx, reportlab), plus LibreOffice for the template PDF.

**Test without a key:** `POLICY_OFFLINE=1 ./run_all.sh` runs everything with an offline stand-in model and writes only to `*_offline` folders.
