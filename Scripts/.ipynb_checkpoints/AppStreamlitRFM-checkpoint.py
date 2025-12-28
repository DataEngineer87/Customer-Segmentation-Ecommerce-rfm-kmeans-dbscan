#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# STREAMLIT RFM DASHBOARD – OLIST MARKETPLACE
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import datetime as dt

st.set_page_config(
    page_title="Segmentation RFM – Olist",
    layout="wide",
    initial_sidebar_state="expanded"
)
import base64

# Convertir l'image en base64
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

img_base64 = get_base64_image("/home/sacko/Documents/SEGMENTATION_ECOMERCE/images/Logo.jpg")

st.markdown(f"""
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
          Alseny Sacko — Data Scientist • Machine Learning Engineer • IA Générative & MLOps
    </div>
""", unsafe_allow_html=True)


# FONCTIONS UTILITAIRES
def load_olist_data(uploaded_file):
    """Charge le dataset Olist et convertit les colonnes de dates."""
    df = pd.read_csv(uploaded_file)

    date_cols = [
        "shipping_limit_date", "order_purchase_timestamp", "order_approved_at",
        "order_delivered_carrier_date", "order_delivered_customer_date",
        "order_estimated_delivery_date", "review_creation_date",
        "review_answer_timestamp"
    ]

    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


def prepare_base_for_segmentation(df):
    """Nettoie et prépare les colonnes utiles pour RFM."""
    df = df.copy()

    if "order_status" in df.columns:
        df = df[df["order_status"] == "delivered"]

    df = df.drop_duplicates(subset=["order_id", "customer_unique_id"])

    needed = [
        "order_id", "customer_unique_id", "order_purchase_timestamp",
        "payment_value", "review_score"
    ]
    df = df[[c for c in needed if c in df.columns]]

    df = df.dropna(subset=["customer_unique_id", "order_purchase_timestamp"])
    return df


# RFM CALCUL
def compute_rfm_table(df):
    """Calcule les métriques RFM."""
    df = df.copy()

    max_ts = df["order_purchase_timestamp"].max()
    NOW = max_ts + pd.Timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg(
        recency=("order_purchase_timestamp", lambda x: (NOW - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary_value=("payment_value", "sum")
    )

    return rfm


def add_rfm_segments(rfm):
    """Ajoute scores RFM + segmentation marketing standard."""
    rfm = rfm.copy()

    # Scores RFM
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary_value"].rank(method="first"), 5, labels=[1,2,3,4,5])

    # Cast en int
    for c in ["recency_score","frequency_score","monetary_score"]:
        rfm[c] = rfm[c].astype(int)

    # Segmentation
    def segment_rule(row):
        r,f,m = row["recency_score"], row["frequency_score"], row["monetary_score"]

        if r>=4 and f>=4 and m>=4:
            return "champions"
        if r>=4 and f>=4:
            return "clients_fideles"
        if r>=4 and f>=3 and m>=3:
            return "clients_prometteurs"
        if r>=4 and f<=2:
            return "nouveaux_clients"
        if r>=3 and f<=2 and m>=2:
            return "clients_a_surveiller"
        if r==3 and f>=3:
            return "clients_a_ne_pas_perdre"
        if r<=2 and (f>=3 or m>=4):
            return "clients_a_reactiver"
        if r<=2 and f<=2 and m<=2:
            return "clients_perdus"
        if r<=2 and f<=2:
            return "clients_en_risque"
        return "autres"

    rfm["segment"] = rfm.apply(segment_rule, axis=1)
    return rfm


SEGMENT_DESCRIPTIONS = {
    "champions": "Clients très récents, très actifs et à forte valeur. Priorité : fidélisation premium.",
    "clients_fideles": "Achètent souvent, réguliers et engagés. Très bonne base pour cross-sell.",
    "clients_prometteurs": "Clients récents à fort potentiel. Objectif : les transformer en champions.",
    "nouveaux_clients": "Clients récents avec peu d’achats. À engager avec une bonne intégration.",
    "clients_a_surveiller": "Récents mais peu actifs. Nécessitent recommandations personnalisées.",
    "clients_a_ne_pas_perdre": "Début de désengagement. Actions préventives recommandées.",
    "clients_a_reactiver": "Clients anciens historiquement rentables. Grande priorité de reconquête.",
    "clients_perdus": "Très faible activité. Cible idéale pour campagnes automatisées low-cost.",
    "clients_en_risque": "Anciens et peu actifs mais valeur non négligeable. Effort modéré possible.",
    "autres": "Profil non classé. Données insuffisantes ou atypiques."
}

# GRAPHES

def treemap_rfm(rfm):
    counts = rfm["segment"].value_counts()
    labels = [f"{s}\n({c})" for s,c in counts.items()]
    colors = sns.color_palette("Spectral", len(counts))

    fig = plt.figure(figsize=(12,8))
    squarify.plot(sizes=counts.values, label=labels, color=colors, alpha=0.85)
    plt.axis("off")
    return fig


def barplot_rfm(rfm):
    counts = rfm["segment"].value_counts()
    fig = plt.figure(figsize=(12,6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.xticks(rotation=45)
    plt.title("Nombre de clients par segment RFM")
    return fig


def pie_rfm(rfm):
    counts = rfm["segment"].value_counts()
    fig = plt.figure(figsize=(8,8))
    plt.pie(counts.values, labels=counts.index, autopct="%1.1f%%", startangle=90)
    plt.title("Répartition des segments RFM")
    plt.axis("equal")
    return fig


# INTERFACE STREAMLIT
st.title("Segmentation RFM – Olist Marketplace")
st.markdown("Analyse interactive des segments clients basée sur la méthode **RFM** (Recency, Frequency, Monetary).")

uploaded_file = st.file_uploader("📂 Importez votre fichier olist_master_dataset.csv", type=["csv"])

if uploaded_file:

    # Préparation
    st.subheader("Préparation des données")
    df_raw = load_olist_data(uploaded_file)
    df_seg = prepare_base_for_segmentation(df_raw)

    st.write("Données chargées :", df_seg.head())

    # Méthode RFM
    st.subheader("Calcul du RFM")
    rfm = compute_rfm_table(df_seg)
    rfm = add_rfm_segments(rfm)

    st.write("Aperçu RFM :", rfm.head())

    # Visualisations
    st.subheader("Segmentation RFM")

    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(treemap_rfm(rfm))
    with col2:
        st.pyplot(barplot_rfm(rfm))

    st.pyplot(pie_rfm(rfm))

    # Interprétation dynamique
    st.subheader("Interprétation des segments")

    selected_segment = st.selectbox("Choisissez un segment :", rfm["segment"].unique())
    st.info(SEGMENT_DESCRIPTIONS.get(selected_segment, "Segment inconnu."))

    # Export
    st.subheader("Export des données segmentées")
    st.download_button(
        label="Télécharger fichier RFM segmenté",
        data=rfm.to_csv().encode(),
        file_name="olist_rfm_segments.csv"
    )

else:
    st.warning("Veuillez importer un fichier CSV pour commencer.")

