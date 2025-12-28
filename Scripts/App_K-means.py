#!/usr/bin/env python
# coding: utf-8

# # STREAMLIT TABLEAU DE BORD - SEGMENTATION CLIENT K-MEANS

# In[2]:


import streamlit as st
import pandas as pd
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import squarify
import base64

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================

st.set_page_config(
    page_title="Segmentation Client – K-Means",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
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
            border-radius: 6px;
        }}
    </style>

    <div class="top-banner">
        <img src="data:image/jpeg;base64,{img_base64}" class="banner-img">
        Alseny Sacko — Data Scientist confirmé orienté MLOps & GenAI
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# CHARGEMENT DES DONNÉES
# ============================================================
st.title("Segmentation Client – K-Means")

uploaded_file = st.file_uploader("📂 Importer df_kmeans.csv", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # ========================================================
    # FEATURES
    # ========================================================
    features = [
        "recency",
        "frequency",
        "monetary_value",
        "mean_review_score",
        "mean_payment_installments"
    ]

    X = df[features]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # =========================================================
    # PARAMÈTRES K
    # ========================================================
    k = st.slider("Nombre de clusters (k)", 2, 8, 4)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil = silhouette_score(X_scaled, labels)

    df["cluster"] = labels

    st.success(f"Silhouette score : {sil:.3f}")

    # =========================================================
    # VISUALISATION INTERACTIVE
    # ==========================================================
    st.subheader("📊 Visualisation des clusters")

    fig = px.scatter_3d(
        df,
        x="recency",
        y="frequency",
        z="monetary_value",
        color="cluster",
        title="Clusters clients – K-Means (RFM)",
        opacity=0.7
    )
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # PROFILING
    # ==============================================================
    st.subheader("Profil des clusters")

    profile = (
        df.groupby("cluster")[features]
        .mean()
        .round(2)
        .reset_index()
    )

    st.dataframe(profile)

    # ==========================================================
    # INTERPRÉTATION AUTOMATIQUE
    # ==========================================================
    st.subheader("Interprétation automatique")

    global_means = df[["recency","frequency","monetary_value"]].mean()

    for _, row in profile.iterrows():
        c = row["cluster"]

        desc = []
        if row["recency"] < global_means["recency"]:
            desc.append("clients récents")
        else:
            desc.append("clients inactifs")

        if row["frequency"] > global_means["frequency"]:
            desc.append("achats fréquents")
        else:
            desc.append("faible fréquence")

        if row["monetary_value"] > global_means["monetary_value"]:
            desc.append("forte valeur")
        else:
            desc.append("faible valeur")

        st.markdown(f"### 🔹 Cluster {int(c)}")
        st.info(", ".join(desc))

    # ===========================================================
    # EXPORTATTION DES SEGMENTS
    # ============================================================
    st.download_button(
        "Exporter les clusters",
        df.to_csv(index=False).encode(),
        file_name="clients_kmeans_clusters.csv"
    )

else:
    st.warning("Veuillez importer un fichier CSV.")


# In[ ]:




