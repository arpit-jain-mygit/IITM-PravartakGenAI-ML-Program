"""
Phase 1 - load, audit and clean the course dataset; build the data dictionary.

Key data-quality findings that shape every later step (shown in notebook 01):
- product_id is NOT a stable product key: each id appears in 5 rows with different category, brand and price.
  We therefore treat every row's product as its own catalogue item and give it a new stable key, item_id
  (ITM0000-ITM1499). The original product_id is kept for traceability.
- customer_id is likewise not stable (6 rows per id with different age / membership), so customer analysis
  is done at the interaction level (one row = one customer-product interaction), not per customer_id.
- size is missing in 270 rows (18%), spread evenly over all categories, so it is not a structural "no size"
  case. It is filled with an explicit "Unknown" category plus a size_missing indicator.
- Post-purchase columns (returned_flag, rating_given, review_sentiment_label, customer_review_text) only
  make sense after a purchase and must never be used to predict purchase_flag (leakage).
"""

import json

import numpy as np
import pandas as pd

from src.config import POLICY_JSON, PROCESSED, RAW_CSV

# ---------------------------------------------------------------------------
# Column groups (original vs engineered) - drives the data dictionary
# ---------------------------------------------------------------------------
GROUPS = {
    "Customer profile": ["customer_id", "age", "gender", "city_tier", "state", "income_band", "occupation",
                         "membership_level"],
    "Customer history": ["days_since_last_purchase", "avg_order_value", "total_orders", "preferred_channel"],
    "Product profile": ["product_id", "category", "sub_category", "brand", "color", "size", "material", "season",
                        "is_returnable"],
    "Commercial data": ["base_price", "discount_percent", "final_price", "avg_product_rating", "stock_level"],
    "Interaction data": ["purchase_flag", "quantity", "payment_mode", "delivery_type", "time_on_page",
                         "added_to_cart", "wishlist_flag"],
    "Post-purchase data": ["returned_flag", "rating_given", "review_sentiment_label"],
    "GenAI text extension": ["product_title", "product_description", "customer_review_text", "customer_query",
                             "policy_doc_id", "policy_text_snippet", "image_path", "recommended_use_case",
                             "style_tags", "occasion_tags"],
}

DESCRIPTIONS = {
    "customer_id": "Customer identifier (NOT stable across rows - see data notes)",
    "age": "Customer age in years", "gender": "Customer gender", "city_tier": "Tier-1/2/3 city",
    "state": "Indian state", "income_band": "Low / Medium / High", "occupation": "Customer occupation",
    "membership_level": "Bronze < Silver < Gold < Platinum loyalty tier",
    "days_since_last_purchase": "Recency: days since the customer's previous order",
    "avg_order_value": "Customer's historical average order value (INR)",
    "total_orders": "Customer's historical number of orders", "preferred_channel": "App / Web / Store",
    "product_id": "Source product identifier (NOT stable - one id maps to 5 different products)",
    "category": "Accessories / Apparel / Shoes", "sub_category": "12 sub-categories, e.g. Sneakers, Watch, Jeans",
    "brand": "Product brand", "color": "Product colour", "size": "S/M/L/XL (18% missing)",
    "material": "Main material", "season": "Season the product is designed for",
    "is_returnable": "1 if the item is eligible for return",
    "base_price": "List price before discount (INR)", "discount_percent": "Discount offered (0-40%)",
    "final_price": "Price after discount (INR)", "avg_product_rating": "Average product rating (1-5)",
    "stock_level": "Units in stock",
    "purchase_flag": "TARGET - 1 if the interaction ended in a purchase",
    "quantity": "Units in the interaction", "payment_mode": "COD / UPI / Card / Wallet",
    "delivery_type": "Standard / Express", "time_on_page": "Seconds spent on the product page",
    "added_to_cart": "1 if the product was added to cart", "wishlist_flag": "1 if wishlisted",
    "returned_flag": "1 if the purchase was returned (post-purchase)", "rating_given": "Rating given by the customer",
    "review_sentiment_label": "positive / neutral / negative label of the review",
    "product_title": "Short product title", "product_description": "Templated product description",
    "customer_review_text": "Review text (28 distinct templates)",
    "customer_query": "Synthetic shopper query for this product", "policy_doc_id": "Policy linked to the row",
    "policy_text_snippet": "Text of the linked policy", "image_path": "Placeholder image path (no images supplied)",
    "recommended_use_case": "Primary use case", "style_tags": "Comma-separated style tags",
    "occasion_tags": "Occasion tag",
}

ENGINEERED = {
    "item_id": ("Key", "Stable unique product key, one per row (ITM0000...)", "row order of the raw file"),
    "size_filled": ("Product profile", "size with missing values set to 'Unknown'", "size.fillna('Unknown')"),
    "size_missing": ("Product profile", "1 if size was missing", "size.isna()"),
    "discount_amount": ("Commercial data", "base_price - final_price (INR saved)", "base_price - final_price"),
    "price_to_aov": ("Commercial data", "Item price relative to the customer's usual basket",
                     "final_price / avg_order_value"),
    "income_ord": ("Customer profile", "Ordinal income: Low=0, Medium=1, High=2", "map(income_band)"),
    "membership_ord": ("Customer profile", "Ordinal membership: Bronze=0 ... Platinum=3", "map(membership_level)"),
    "engagement": ("Interaction data", "added_to_cart + wishlist_flag (0-2)", "added_to_cart + wishlist_flag"),
    "price_band": ("Commercial data", "Budget <1500 / Mid 1500-3500 / Premium >3500 (final_price)", "pd.cut"),
}

INCOME = {"Low": 0, "Medium": 1, "High": 2}
MEMBERSHIP = {"Bronze": 0, "Silver": 1, "Gold": 2, "Platinum": 3}


def load_raw() -> pd.DataFrame:
    return pd.read_csv(RAW_CSV)


def load_policies() -> dict:
    return json.loads(POLICY_JSON.read_text())


def audit(df: pd.DataFrame) -> dict:
    """Facts used in notebook 01 to justify the cleaning decisions."""
    by_pid = df.groupby("product_id")
    by_cid = df.groupby("customer_id")
    return {
        "rows": len(df), "columns": df.shape[1],
        "missing": df.isna().sum()[lambda s: s > 0].to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "rows_per_product_id": int(by_pid.size().iloc[0]),
        "product_ids_with_conflicting_category": int((by_pid["category"].nunique() > 1).sum()),
        "product_ids_with_conflicting_brand": int((by_pid["brand"].nunique() > 1).sum()),
        "rows_per_customer_id": int(by_cid.size().iloc[0]),
        "customer_ids_with_conflicting_age": int((by_cid["age"].nunique() > 1).sum()),
        "size_missing_share_by_category": df.assign(m=df["size"].isna()).groupby("category")["m"].mean().round(3).to_dict(),
        "purchase_rate": round(df["purchase_flag"].mean(), 3),
        "returned_given_not_purchased": int(((df.purchase_flag == 0) & (df.returned_flag == 1)).sum()),
        "rating5_given_not_purchased": int(((df.purchase_flag == 0) & (df.rating_given == 5)).sum()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.insert(0, "item_id", [f"ITM{i:04d}" for i in range(len(out))])
    out["size_missing"] = out["size"].isna().astype(int)
    out["size_filled"] = out["size"].fillna("Unknown")
    out["discount_amount"] = (out["base_price"] - out["final_price"]).round(2)
    out["price_to_aov"] = (out["final_price"] / out["avg_order_value"]).round(4)
    out["income_ord"] = out["income_band"].map(INCOME)
    out["membership_ord"] = out["membership_level"].map(MEMBERSHIP)
    out["engagement"] = out["added_to_cart"] + out["wishlist_flag"]
    out["price_band"] = pd.cut(out["final_price"], [0, 1500, 3500, np.inf], labels=["Budget", "Mid", "Premium"]).astype(str)
    # the product_title template drops the final "s" of Accessories ("...Accessorie"); fix it for retrieval text
    out["product_title"] = out["product_title"].str.replace(r"Accessorie$", "Accessories", regex=True)
    return out


def data_dictionary(clean_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group, cols in GROUPS.items():
        for c in cols:
            s = clean_df[c]
            example = s.dropna().iloc[0]
            rows.append({"column": c, "group": group, "origin": "original (course dataset)",
                         "dtype": str(s.dtype), "missing": int(s.isna().sum()), "unique": int(s.nunique()),
                         "example": str(example)[:60], "description": DESCRIPTIONS.get(c, ""), "how_created": ""})
    for c, (group, desc, how) in ENGINEERED.items():
        s = clean_df[c]
        rows.append({"column": c, "group": group, "origin": "engineered (this project)", "dtype": str(s.dtype),
                     "missing": int(s.isna().sum()), "unique": int(s.nunique()), "example": str(s.iloc[0])[:60],
                     "description": desc, "how_created": how})
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    """Reproduce data/processed/* from the raw files."""
    raw = load_raw()
    df = clean(raw)
    df.to_csv(PROCESSED / "retail_clean.csv", index=False)
    data_dictionary(df).to_csv(PROCESSED / "data_dictionary.csv", index=False)
    return df


def load_clean() -> pd.DataFrame:
    path = PROCESSED / "retail_clean.csv"
    return pd.read_csv(path) if path.exists() else run()
