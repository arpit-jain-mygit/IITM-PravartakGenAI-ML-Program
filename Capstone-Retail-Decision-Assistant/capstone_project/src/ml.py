"""
Phase 2 - classical ML.

Supervised: predict purchase_flag (will this product view convert?) from features known BEFORE checkout.
Unsupervised: K-Means personas on customer-history + price features, profiled in business language,
with a price-sensitivity score per segment (used by the assistant's analytics tool).
"""

import json

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import OUTPUTS, SEED

TARGET = "purchase_flag"

NUM_FEATURES = ["age", "days_since_last_purchase", "avg_order_value", "total_orders", "base_price",
                "discount_percent", "final_price", "avg_product_rating", "stock_level", "time_on_page",
                "discount_amount", "price_to_aov", "income_ord", "membership_ord", "engagement"]
BIN_FEATURES = ["added_to_cart", "wishlist_flag", "is_returnable", "size_missing"]
CAT_FEATURES = ["gender", "city_tier", "state", "occupation", "preferred_channel", "category", "sub_category",
                "brand", "color", "size_filled", "material", "season"]

# Columns deliberately NOT used to predict purchase_flag, with the reason (shown in notebook 02)
EXCLUDED = {
    "returned_flag": "post-purchase outcome - only possible after a purchase (target leakage)",
    "rating_given": "post-purchase; rating 5 appears only for purchasers (leakage)",
    "review_sentiment_label": "post-purchase review label (leakage)",
    "customer_review_text": "post-purchase free text (leakage)",
    "payment_mode": "chosen at checkout, i.e. at the moment of purchase (not known when predicting)",
    "delivery_type": "chosen at checkout (not known when predicting)",
    "quantity": "basket quantity is set at checkout",
    "customer_id, product_id": "identifiers that are not stable keys in this dataset",
    "income_band, membership_level": "replaced by their ordinal versions income_ord / membership_ord",
    "size": "replaced by size_filled + size_missing",
    "product_title / description / tags / query / policy columns": "text used by the GenAI part, not the tabular model",
}


def preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))] + ([("scale", StandardScaler())] if scale_numeric else [])
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_FEATURES),
        ("bin", SimpleImputer(strategy="most_frequent"), BIN_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False))]),
         CAT_FEATURES),
    ])


def candidate_models() -> dict:
    """name -> (pipeline, small hyper-parameter grid)."""
    return {
        "Dummy (majority)": (Pipeline([("prep", preprocessor()), ("clf", DummyClassifier(strategy="prior"))]), {}),
        "Logistic Regression": (Pipeline([("prep", preprocessor()),
                                          ("clf", LogisticRegression(max_iter=3000, solver="liblinear"))]),
                                {"clf__C": [0.01, 0.03, 0.1, 0.3, 1.0], "clf__penalty": ["l1", "l2"]}),
        "Random Forest": (Pipeline([("prep", preprocessor(False)),
                                    ("clf", RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1))]),
                          {"clf__max_depth": [4, 6, 10, None], "clf__min_samples_leaf": [5, 10, 20],
                           "clf__max_features": ["sqrt", 0.3]}),
        "Gradient Boosting (HGB)": (Pipeline([("prep", preprocessor(False)),
                                              ("clf", HistGradientBoostingClassifier(random_state=SEED))]),
                                    {"clf__learning_rate": [0.03, 0.1], "clf__max_depth": [2, 3, 4],
                                     "clf__max_iter": [100, 300], "clf__l2_regularization": [0.0, 1.0]}),
    }


# ---------------------------------------------------------------------------
# Unsupervised: RFM customer segments (Recency, Frequency, Monetary value)
# ---------------------------------------------------------------------------
# RFM is the standard retail segmentation. In this dataset it also separates far better than a wide
# mixed feature set (silhouette ~0.26 vs ~0.09, compared in notebook 02), because the synthetic
# features are almost uncorrelated. Other attributes are used afterwards to PROFILE the segments.
SEG_FEATURES = ["days_since_last_purchase", "total_orders", "avg_order_value"]
ALT_SEG_FEATURES = ["age", "income_ord", "membership_ord", "avg_order_value", "total_orders",
                    "days_since_last_purchase", "final_price", "discount_percent", "time_on_page"]


def kmeans_diagnostics(X_scaled: np.ndarray, ks=range(2, 9)) -> pd.DataFrame:
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X_scaled)
        rows.append({"k": k, "inertia": km.inertia_,
                     "silhouette": silhouette_score(X_scaled, km.labels_, sample_size=None, random_state=SEED),
                     "smallest_cluster_share": np.bincount(km.labels_).min() / len(km.labels_)})
    return pd.DataFrame(rows)


def fit_segments(df: pd.DataFrame, k: int):
    scaler = StandardScaler().fit(df[SEG_FEATURES])
    Xs = scaler.transform(df[SEG_FEATURES])
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(Xs)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward").fit(Xs)
    ari = adjusted_rand_score(km.labels_, agg.labels_)
    return km, scaler, Xs, ari


def price_sensitivity(df: pd.DataFrame, labels) -> pd.DataFrame:
    """Per segment: purchase rate on deep-discount items (>=25%) minus rate on low-discount items (<=10%).
    A larger 'discount lift' means the segment converts much more when there is a discount."""
    d = df.assign(segment=labels)
    hi = d[d.discount_percent >= 25].groupby("segment")["purchase_flag"].mean()
    lo = d[d.discount_percent <= 10].groupby("segment")["purchase_flag"].mean()
    return pd.DataFrame({"conv_deep_discount": hi, "conv_low_discount": lo, "discount_lift": hi - lo}).round(3)


def name_segments(profile: pd.DataFrame) -> dict:
    """Name each segment from its RFM position relative to the average (z-scores of segment means).
    Greedy: the best-matching segment takes each name, so names are unique and reproducible."""
    z = (profile[["days_since_last_purchase", "total_orders", "avg_order_value"]] -
         profile[["days_since_last_purchase", "total_orders", "avg_order_value"]].mean()) / \
        profile[["days_since_last_purchase", "total_orders", "avg_order_value"]].std(ddof=0)
    z.columns = ["R", "F", "M"]          # R = days since last purchase (higher = less recent)
    rules = [
        ("High-Value Regulars", lambda r: r.F + r.M - r.R),
        ("Lapsing Customers", lambda r: r.R - r.F),
        ("Frequent Small-Basket Buyers", lambda r: r.F - r.M),
        ("Occasional Buyers", lambda r: -r.F),
        ("Recent Low-Value Shoppers", lambda r: -r.R - r.M),
        ("Average Shoppers", lambda r: -abs(r).sum()),
    ]
    names, left = {}, list(z.index)
    for name, score in rules:
        if not left:
            break
        seg = max(left, key=lambda s: score(z.loc[s]))
        names[seg] = name
        left.remove(seg)
    return names


def segment_profile(df: pd.DataFrame, labels) -> pd.DataFrame:
    d = df.assign(segment=labels)
    prof = d.groupby("segment").agg(
        size=("item_id", "count"), age=("age", "mean"), income_ord=("income_ord", "mean"),
        membership_ord=("membership_ord", "mean"), avg_order_value=("avg_order_value", "mean"),
        total_orders=("total_orders", "mean"), days_since_last_purchase=("days_since_last_purchase", "mean"),
        final_price=("final_price", "mean"), discount_percent=("discount_percent", "mean"),
        time_on_page=("time_on_page", "mean"), purchase_rate=("purchase_flag", "mean"),
        return_rate=("returned_flag", "mean"))
    prof["share"] = prof["size"] / prof["size"].sum()
    prof = prof.join(price_sensitivity(df, labels))
    return prof.round(3)


def save_segment_summary(profile: pd.DataFrame, names: dict, path=None):
    """Written by notebook 02, read by the assistant's analytics tool."""
    path = path or OUTPUTS / "segment_summary.json"
    out = {"segments": [{"segment": int(s), "name": names[s], **{k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                                                                 for k, v in row.items()}}
                        for s, row in profile.iterrows()],
           "method": "K-Means on standardised customer-history and price features; price sensitivity = "
                     "conversion on >=25% discount minus conversion on <=10% discount"}
    path.write_text(json.dumps(out, indent=2))
    return out
