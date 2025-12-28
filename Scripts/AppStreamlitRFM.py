#!/usr/bin/env python
# coding: utf-8

# # STREAMLIT RFM TABLEAU DE BORD - OLIST MARKETPLACE

# In[2]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import squarify
import base64

# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Segmentation RFM – Olist",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================
# EN-TÊTE AVEC IMAGE
# ============================================================
#image_path = "/home/sacko/Documents/SEGMENTATION_ECOMERCE/images/Logo.jpg"
image_path = "images/Logo.jpg"
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

# ==============================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ==============================================================
def load_data(file):
    df = pd.read_csv(file)

    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date"
    ]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    return df


def prepare_base(df):
    df = df.copy()

    if "order_status" in df.columns:
        df = df[df["order_status"] == "delivered"]

    df = df.drop_duplicates(subset=["order_id", "customer_unique_id"])
    cols_needed = [
        "order_id",
        "customer_unique_id",
        "order_purchase_timestamp",
        "payment_value"
    ]

    df = df[cols_needed]
    df = df.dropna()

    return df


# ===============================================================
# RFM CALCULATION
# ================================================================
def compute_rfm(df):
    max_date = df["order_purchase_timestamp"].max()
    now = max_date + pd.Timedelta(days=1)

    rfm = df.groupby("customer_unique_id").agg(
        recency=("order_purchase_timestamp", lambda x: (now - x.max()).days),
        frequency=("order_id", "nunique"),
        monetary_value=("payment_value", "sum")
    )

    return rfm

def add_rfm_segments(rfm):
    rfm = rfm.copy()

    # Scores
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels=[5,4,3,2,1]).astype(int)
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)
    rfm["monetary_score"] = pd.qcut(rfm["monetary_value"].rank(method="first"), 5, labels=[1,2,3,4,5]).astype(int)

    rfm["rfm_score"] = (
        rfm["recency_score"].astype(str) +
        rfm["frequency_score"].astype(str) +
        rfm["monetary_score"].astype(str)
    )

    def segment_rule(r, f, m):
        if r>=4 and f>=4 and m>=4: return "champions"
        if r>=4 and f>=4: return "clients_fideles"
        if r>=4 and f>=3 and m>=3: return "clients_prometteurs"
        if r>=4 and f<=2: return "nouveaux_clients"
        if r>=3 and f<=2 and m>=2: return "clients_a_surveiller"
        if r==3 and f>=3: return "clients_a_ne_pas_perdre"
        if r<=2 and (f>=3 or m>=4): return "clients_a_reactiver"
        if r<=2 and f<=2 and m<=2: return "clients_perdus"
        if r<=2 and f<=2: return "clients_en_risque"
        return "autres"

    rfm["segment"] = rfm.apply(
        lambda x: segment_rule(
            x["recency_score"],
            x["frequency_score"],
            x["monetary_score"]
        ),
        axis=1
    )

    return rfm

# ===============================================================
# DESCRIPTIONS MÉTIER
# ================================================================
SEGMENT_DESCRIPTIONS = {
    "champions": "Clients récents, très actifs et à forte valeur. Fidélisation premium.",
    "clients_fideles": "Clients réguliers et engagés. Cross-sell & upsell.",
    "clients_prometteurs": "Clients récents avec potentiel élevé.",
    "nouveaux_clients": "Clients récents à activer avec onboarding.",
    "clients_a_surveiller": "Potentiel latent, faible engagement.",
    "clients_a_ne_pas_perdre": "Début de désengagement.",
    "clients_a_reactiver": "Clients anciens mais historiquement rentables.",
    "clients_perdus": "Très faible activité.",
    "clients_en_risque": "Valeur correcte mais inactifs.",
    "autres": "Profils atypiques."
}

# ================================================================
# VISUALISATIONS
# =================================================================

def barplot_rfm(rfm):
    counts = rfm["segment"].value_counts()
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis", ax=ax)

    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v} ({v/total:.1%})", ha="center", va="bottom")

    ax.set_title("Nombre de clients par segment RFM")
    ax.set_xlabel("Segment")
    ax.set_ylabel("Nombre de clients")
    plt.xticks(rotation=45)
    return fig

def treemap_rfm(rfm):
    counts = rfm["segment"].value_counts()
    fig = plt.figure(figsize=(12,8))
    squarify.plot(
        sizes=counts.values,
        label=[f"{s}\n({c})" for s,c in counts.items()],
        color=sns.color_palette("Spectral", len(counts)),
        alpha=0.85
    )
    plt.axis("off")
    return fig

# =====================================================================
# INTERFACE STREAMLIT
# ======================================================================

st.title("Segmentation RFM – Olist Marketplace")

uploaded_file = st.file_uploader("📂 Importez votre fichier CSV", type=["csv"])

if uploaded_file:

    df_raw = load_data(uploaded_file)
    df_base = prepare_base(df_raw)

    rfm = compute_rfm(df_base)
    rfm = add_rfm_segments(rfm)

    st.subheader("📊 Segmentation RFM")
    st.dataframe(rfm.head())

    # Filtres
    segments_selected = st.multiselect(
        "Sélectionnez les segments",
        sorted(rfm["segment"].unique()),
        default=sorted(rfm["segment"].unique())
    )

    rfm_filtered = rfm[rfm["segment"].isin(segments_selected)]

    st.write(f"**{len(rfm_filtered)} clients sélectionnés**")

    # ==============================================================
    # Visualisation
    # ===============================================================
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(barplot_rfm(rfm_filtered))
    with col2:
        st.pyplot(treemap_rfm(rfm_filtered))

    # ===============================================================   
    # Interprétatio
    # ================================================================
    st.subheader("Interprétation dynamique")

    selected_segment = st.selectbox(
        "Choisissez un segment",
        sorted(rfm_filtered["segment"].unique())
    )

    st.info(SEGMENT_DESCRIPTIONS[selected_segment])

    stats = rfm_filtered[rfm_filtered["segment"] == selected_segment][
        ["recency", "frequency", "monetary_value"]
    ].mean().round(2)

    st.markdown("### Indicateurs moyens")
    st.metric("Récence moyenne (jours)", int(stats["recency"]))
    st.metric("Fréquence moyenne", stats["frequency"])
    st.metric("Valeur monétaire moyenne", stats["monetary_value"])

    # ===================================================
    # Exportation des segments
    #=====================================================
    st.subheader("Export")
    st.download_button(
        "Télécharger les données RFM",
        data=rfm_filtered.to_csv().encode(),
        file_name="olist_rfm_segments.csv"
    )

else:
    st.warning("Veuillez importer un fichier CSV pour commencer.")


# In[ ]:




