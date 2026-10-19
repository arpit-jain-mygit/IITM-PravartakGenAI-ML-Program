"""Regression-like checks for the data pipeline, ML artefacts, retrieval, grounding validators and routing."""
import json

import pytest

from src import config, data_prep, documents


@pytest.fixture(scope="session", autouse=True)
def prepared():
    df = data_prep.run()
    documents.write_all(df)
    if not (config.OUTPUTS / "segment_summary.json").exists():
        from src import ml
        km, *_ = ml.fit_segments(df, 4)
        prof = ml.segment_profile(df, km.labels_)
        ml.save_segment_summary(prof, ml.name_segments(prof))
    return df


# ---------------- data ----------------
def test_clean_data_contract(prepared):
    df = prepared
    assert len(df) == 1500 and df["item_id"].is_unique
    assert df["size_filled"].isna().sum() == 0 and df["size_missing"].sum() == 270
    assert set(df["price_band"]) <= {"Budget", "Mid", "Premium"}


def test_no_leakage_features():
    from src import ml
    used = set(ml.NUM_FEATURES + ml.BIN_FEATURES + ml.CAT_FEATURES)
    assert not used & {"returned_flag", "rating_given", "review_sentiment_label", "payment_mode", "delivery_type", "quantity"}


# ---------------- retrieval + recommendation ----------------
def test_constraint_parsing():
    from src.recommender import parse_constraints
    c = parse_constraints("Recommend casual sneakers under ₹4000 for daily wear.")
    assert c["max_price"] == 4000 and c["sub_category"] == ["Sneakers"]
    c = parse_constraints("premium customer who prefers black formal shoes")
    assert c["color"] == ["Black"] and c["sub_category"] == ["Formal"] and c["premium"]


def test_recommendations_respect_budget_and_category():
    from src.recommender import recommend
    r = recommend("Recommend casual sneakers under ₹4000 for daily wear.")
    assert r["recommended_products"]
    ids = {x["item_id"] for x in r["retrieved"]}
    meta = {d["doc_id"]: d["meta"] for d in documents.load_product_docs()}
    for p in r["recommended_products"]:
        assert p["item_id"] in ids and p["price"] <= 4000 and meta[p["item_id"]]["sub_category"] == "Sneakers"


def test_validator_drops_hallucinated_and_over_budget_items():
    from src.recommender import validate
    hits = [{"doc_id": "ITM0001", "meta": {"final_price": 1000, "product_title": "A"}},
            {"doc_id": "ITM0002", "meta": {"final_price": 5000, "product_title": "B"}}]
    ans = {"recommended_products": [{"item_id": "ITM0001"}, {"item_id": "ITM0002"}, {"item_id": "ITM9999"}]}
    out, issues = validate(ans, hits, {"max_price": 4000})
    assert [p["item_id"] for p in out["recommended_products"]] == ["ITM0001"] and len(issues) == 2


def test_no_match_is_graceful():
    from src.recommender import recommend
    r = recommend("Recommend a red watch under ₹100")
    assert r["status"] == "no_match" and not r["recommended_products"]


# ---------------- policy QA ----------------
@pytest.mark.parametrize("q,doc", [("Can I return a discounted product after 7 days?", "POL_RETURNS"),
                                   ("How long does express delivery take?", "POL_SHIPPING"),
                                   ("Is physical damage to my watch covered under warranty?", "POL_WARRANTY"),
                                   ("Can I exchange for a different size?", "POL_EXCHANGE"),
                                   ("When will my prepaid refund arrive?", "POL_REFUND")])
def test_policy_answers_cite_the_right_document(q, doc):
    from src.policy_qa import answer_policy
    r = answer_policy(q)
    assert doc in r["citations"] and r["quotes"]


def test_policy_refuses_when_not_covered():
    from src.policy_qa import answer_policy
    r = answer_policy("Can I pay with cryptocurrency?")
    assert r["status"] == "not_covered" and not r["citations"]


def test_policy_validator_rejects_invented_quote():
    from src.policy_qa import validate
    ev = [{"doc_id": "POL_RETURNS", "full_text": data_prep.load_policies()["POL_RETURNS"]}]
    bad = {"citations": ["POL_RETURNS"], "quotes": ["Returns are allowed within 30 days"]}
    assert any("verbatim" in i for i in validate(bad, ev))


# ---------------- controller ----------------
@pytest.mark.parametrize("q,tools", [
    ("Recommend casual sneakers under ₹4000 for daily wear.", ["product_search"]),
    ("Can I return a discounted product after 7 days?", ["policy_qa"]),
    ("Is physical damage to my watch covered under the warranty?", ["policy_qa"]),
    ("Which customer segment seems most price-sensitive?", ["segment_insights"]),
    ("Show me a blue backpack under 3000 and tell me if I can exchange it later.", ["product_search", "policy_qa"]),
    ("What do customers say about Nike running shoes?", ["review_summary"]),
    ("What is 15% off 3999?", ["calculator"]),
    ("What's the weather in Mumbai?", []),
])
def test_routing(q, tools):
    from src.agent import route
    assert route(q)[0] == tools


def test_tool_calls_are_bounded():
    from src.agent import RetailAssistant
    from src.config import MAX_TOOL_CALLS
    out = RetailAssistant().ask("recommend sneakers under 3000, can I return them, what do reviews say, "
                                "which segment is price-sensitive and what is 10% off 2000?", verbose=False)
    assert len(out["tools"]) <= MAX_TOOL_CALLS


def test_full_suite_offline_pass_rate():
    from src.evaluation import run_suite
    df = run_suite(verbose=False)
    assert df["PASS"].mean() >= 0.9, df[~df.PASS][["id", "failed_checks"]].to_string()
