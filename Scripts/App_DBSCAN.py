#!/usr/bin/env python
# coding: utf-8

# # STREAMLIT TABLEAU DE BORD - DBSCAN CLIENTS ATYPIQUES

# In[ ]:


import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import squarify
import base64
# ========================================================================
# CONFIGURATION STREAMLIT
# =====================================================================

st.set_page_config(
    page_title="Détection des clients atypiques – DBSCAN",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================================================================
# EN-TÊTE AVEC IMAGE
# ============================================================
image_path = "/home/sacko/Documents/SEGMENTATION_ECOMERCE/images/Logo.jpg"
with open(image_path, "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")

st.markdown(
    f"""
    <style>
        .top-banner {{
            background-color: #0E76A8;
            padding: 12px;
            border-radius: 8px;
            color: white;
            font-size: 22px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        .banner-img {{
            height: 50px;
            border-radius: 6px;TABLEAU DE BORD
        }}
    </style>

    <div class="top-banner">
        <img src="data:image/jpeg;base64,{img_base64}" class="banner-img">
        Alseny Sacko — Data Scientist confirmé orienté MLOps & GenAI
    </div>
    """,
    unsafe_allow_html=True
)

# =================================================================
# DONNEES
# ===================================================================

st.title("Détection des clients atypiques – DBSCAN")
uploaded_file = st.file_uploader("📂 Importer df_dbscan.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    features = [
        "recency",
        "frequency",
        "monetary_value",
        "mean_review_score",
        "mean_payment_installments"
    ]

    X = df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # =============================================================
    # PARAMÈTRES DBSCAN
    # =============================================================
    eps = st.slider("Epsilon (ε)", 0.1, 3.0, 0.8, 0.1)
    min_samples = st.slider("Min samples", 10, 500, 100)

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    df["dbscan_cluster"] = labels

    nb_outliers = (labels == -1).sum()

    st.warning(f"Clients atypiques détectés : {nb_outliers}")

    # ==================================================================
    # VISUALISATION
    # ====================================================================
    st.subheader("📊 Visualisation DBSCAN")

    fig = px.scatter(
        df,
        x="recency",
        y="monetary_value",
        color=df["dbscan_cluster"].astype(str),
        title="DBSCAN – Détection anomalies",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

    # =============================================================
    # CLIENTS ATYPIQUES
    # ===============================================================
    st.subheader("Clients atypiques (actionnables)")

    anomalies = df[df["dbscan_cluster"] == -1]
    st.dataframe(anomalies.head(50))

    # ==================================================================
    # EXPORT
    # ==================================================================
    st.download_button(
        "Exporter clients atypiques",
        anomalies.to_csv(index=False).encode(),
        file_name="clients_atypiques_dbscan.csv"
    )

else:
    st.warning("Veuillez importer un fichier CSV.")

