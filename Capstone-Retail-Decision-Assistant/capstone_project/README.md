# Intelligent Retail Decision Assistant (Capstone)

IITM Pravartak - Professional Certificate Programme in Generative AI and Machine Learning

This is one coherent retail AI prototype. It starts from tabular customer-product interactions, adds classical ML (conversion prediction and customer segmentation), turns the same data into retrieval-ready text for **grounded product recommendations** and **policy QA with citations**, and puts a simple, **traceable controller** in front that routes each request to the right tool.

```
retail_interactions.csv ─┬─> 01 data prep ──> retail_clean.csv + data dictionary
retail_policies.json ────┤
                         ├─> 02 classical ML ──> purchase model (LogReg) + RFM segments ──> segment_summary.json
                         └─> 03/04 documents ──> embeddings ──> FAISS (products, policy sentences)
                                                                   │
user query ─> 05 controller: guardrails → rule-based intents (→ LLM fallback) → ≤3 tools ─┤
              tools: product_search · policy_qa · segment_insights · review_summary ·    │
                     summarise_products · calculator   (each validated, each traced)  <──┘
```

## Project structure

```
capstone_project/
├── data/
│   ├── raw/retail_interactions.csv        course dataset (1,500 rows × 46 columns)
│   ├── policies/retail_policies.json      the 5 course policies (POL_RETURNS, _EXCHANGE, _SHIPPING, _REFUND, _WARRANTY)
│   └── processed/                         retail_clean.csv, data_dictionary.csv, product_docs.jsonl, policy_docs.jsonl
├── notebooks/
│   ├── 01_data_preparation.ipynb          business framing, audit, cleaning, data dictionary
│   ├── 02_classical_ml.ipynb              supervised (purchase_flag) + unsupervised (RFM K-Means)
│   ├── 03_rag_recommendation.ipynb        product documents, FAISS, retrieval inspection, grounded JSON recommendations
│   ├── 04_policy_qa.ipynb                 policy retrieval, cited and quoted answers, refusals
│   ├── 05_agentic_assistant.ipynb         controller, routing table, full demo with traces
│   └── 06_evaluation.ipynb                ML summary, 13 curated queries with regression checks, pytest
├── src/                                   config, data_prep, ml, documents, llm, retrieval, recommender, policy_qa, tools, agent, evaluation
├── tests/test_regressions.py              23 offline regression tests
├── outputs/                               charts, tables, model, segment summary, eval logs, Capstone_Report.pdf
├── README.md, requirements.txt, .env.example
```

## Dataset: source, reproduction and data dictionary

- **Source:** the course-provided `retail_interactions.csv` and `retail_policies.json`. These are the course's supporting files, included in `data/` and in the submission zip.
- **Text columns already in the source.** The GenAI extension columns come with the dataset and were not created by me: `product_title`, `product_description`, `customer_review_text`, `customer_query`, `policy_doc_id`, `policy_text_snippet`, `style_tags`, `occasion_tags`, `recommended_use_case`, `image_path`.
- **Engineered columns (added by me):**
  - `item_id`
  - `size_filled`, `size_missing`
  - `discount_amount`, `price_to_aov`
  - `income_ord`, `membership_ord`
  - `engagement`
  - `price_band`

  They are reproduced by `src/data_prep.py` (`data_prep.run()`, called in notebook 01).
- **Data dictionary:** `data/processed/data_dictionary.csv` gives, for every column, its group, whether it is original or engineered, dtype, missing count, number of unique values, an example, a description and how it was created.
- **Key assumptions** (evidence in notebook 01):
  - `product_id` and `customer_id` are **not stable keys**, so each row is treated as its own catalogue item and analysis is interaction-level.
  - `size` is missing at random (18%), so it is set to `Unknown` with an indicator.
  - Post-purchase and checkout columns are excluded from the purchase model to avoid leakage.
  - No policy text was invented.

## Setup and run

**1. Install the packages.** Python 3.10+, tested on 3.12:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**2. Add your API key.** Copy `.env.example` to `.env` and set `OPENAI_API_KEY`. For the Vocareum lab key, also set `OPENAI_BASE_URL`.

**3. Run the tests and the notebooks in order:**

```bash
python -m pytest -q tests
cd notebooks && for nb in 01_data_preparation 02_classical_ml 03_rag_recommendation 04_policy_qa 05_agentic_assistant 06_evaluation; do jupyter nbconvert --to notebook --execute --inplace "$nb.ipynb"; done
```

Each notebook saves its charts and tables to `outputs/`, and notebook 06 runs the 13 curated test queries.

**Offline mode.** Set `RETAIL_OFFLINE=1` before running the notebooks to run the whole project with no key. It uses local MiniLM embeddings and a deterministic extractive generator instead of the LLM, and writes to `outputs_offline/`. The ML parts are identical in both modes.

**Google Colab:**
1. Upload the folder and run `pip install -r requirements.txt`.
2. Set the keys with `os.environ[...]`.
3. Run the notebooks in order.

## Example queries the assistant handles

| Query | Route |
|---|---|
| Recommend casual sneakers under ₹4000 for daily wear. | product_search (budget + sub-category filters) |
| Can I return a discounted product after 7 days? | policy_qa → cites POL_RETURNS with a verbatim quote |
| Suggest products for a premium customer who prefers black formal shoes. | product_search (colour + sub-category + premium boost) |
| Which customer segment seems most price-sensitive? | segment_insights (from the RFM model, no LLM) |
| Summarise the top three retrieved products and explain why they fit the query. | summarise_products (session memory) |
| Show me a blue backpack under 3000 and tell me if I can exchange it later. | product_search + policy_qa (multi-intent, split clauses) |
| What do customers say about Nike running shoes? | review_summary |
| Is physical damage to my watch covered under the warranty? | policy_qa only ("watch" here is not a shopping request) |
| What's the weather in Mumbai? | guardrail refusal, no tool call |

## Findings (summary; details in the notebooks and `outputs/Capstone_Report.pdf`)

**Supervised: `purchase_flag`**
- Logistic Regression (L1): cross-validated ROC-AUC about 0.64, test ROC-AUC about 0.68. The tree models are no better.
- Conversion is driven by **membership tier**, then income band, product rating and engagement.
- Price and discount barely matter *on average*.

**Unsupervised: RFM K-Means (k = 4, silhouette about 0.26)**
- The segments are High-Value Regulars, Frequent Small-Basket Buyers, Lapsing Customers and Occasional Buyers.
- **Frequent Small-Basket Buyers are the most price-sensitive.** Their conversion rises about 14 points on deep discounts, while the other segments barely react.

**RAG**
- Product retrieval: sub-category precision@5 = 0.93, colour precision@5 = 1.00, and the exact source item in the top 5 for 66% of queries (OpenAI text-embedding-3-small, 300 dataset queries).
- Policy retrieval: hit@1 = 100% on 10 labelled questions. The refusal threshold (0.32) is calibrated between the weakest correct match (0.345) and the strongest out-of-scope question (0.296).
- Every recommendation and policy answer is validated against what was retrieved.

**Evaluation:** 13 curated queries with automatic checks (route, budget, attributes, grounding, citation, verbatim quotes, refusal), plus 23 pytest regression tests.

**Limitations**
- The tabular signal is modest.
- Segments are only moderately separated.
- Review text is templated.
- Source ids are not stable keys.
- No image files are supplied, so the optional multimodal extension was not done.
