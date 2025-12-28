import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score


# ================================================================
# RFM
# ==============================================================
def compute_rfm_table(df: pd.DataFrame) -> pd.DataFrame:
    max_ts = df["order_purchase_timestamp"].max()
    now = max_ts + pd.Timedelta(days=1)

    rfm = (
        df.groupby("customer_unique_id")
        .agg(
            recency=("order_purchase_timestamp", lambda x: (now - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary_value=("payment_value", "sum"),
        )
    )
    return rfm


def prepare_rfm_features(rfm: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(
        rfm[["recency", "frequency", "monetary_value"]]
    )


def fit_kmeans(X: np.ndarray, n_clusters: int):
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    labels = model.fit_predict(X)
    return model, labels


# ================================================================
# MONITORING ARI (fonction attendue par Streamlit)
# =================================================================
def monitor_ari(
    df_transactions: pd.DataFrame,
    k: int = 4,
    window_days: int = 365,
    step_days: int = 7,
):
    """
    Monitoring temporel de la stabilité K-Means via l'ARI.

    Returns
    -------
    days : list[int]
    ari_scores : list[float]
    latest_ari : float
    """

    max_date = df_transactions["order_purchase_timestamp"].max()
    ref_cutoff = max_date - pd.Timedelta(days=window_days)

    # Jeu de référence
    df_ref = df_transactions[
        df_transactions["order_purchase_timestamp"] <= ref_cutoff
    ]

    rfm_ref = compute_rfm_table(df_ref)
    X_ref = prepare_rfm_features(rfm_ref)
    _, labels_ref = fit_kmeans(X_ref, k)

    ari_scores = []
    days = []

    for i in range(0, window_days, step_days):
        df_temp = df_transactions[
            df_transactions["order_purchase_timestamp"]
            <= ref_cutoff + pd.Timedelta(days=i)
        ]

        rfm_temp = compute_rfm_table(df_temp)
        X_temp = prepare_rfm_features(rfm_temp)
        _, labels_temp = fit_kmeans(X_temp, k)

        common_ids = rfm_ref.index.intersection(rfm_temp.index)

        ari = adjusted_rand_score(
            labels_ref[rfm_ref.index.get_indexer(common_ids)],
            labels_temp[rfm_temp.index.get_indexer(common_ids)],
        )

        ari_scores.append(ari)
        days.append(i)

    return days, ari_scores, ari_scores[-1]
