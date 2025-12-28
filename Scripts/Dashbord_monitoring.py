#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
from pathlib import Path

# ============================================================
# Fix import "src" (Streamlit / scripts)
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # racine du projet
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Imports
# ============================================================
import streamlit as st
import pandas as pd
import plotly.express as px
from src.monitoring import monitor_ari
import base64
# ============================================================
# CONFIGURATION STREAMLIT
# ============================================================
st.set_page_config(page_title="Monitoring K-Means", layout="wide")
st.title("Monitoring K-Means – ARI")

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
# Data loading
# ============================================================
DATA_PATH = PROJECT_ROOT / "Donnees" / "df_seg_base.csv"  # ✅ robuste
df = pd.read_csv(DATA_PATH, parse_dates=["order_purchase_timestamp"])

# ============================================================
# Controls
# ============================================================
k = st.slider("Nombre de clusters (k)", 2, 8, 4)
window_days = st.slider("Fenêtre historique (jours)", 180, 730, 365)
step_days = st.slider("Pas temporel (jours)", 7, 30, 7)

# ============================================================
# Monitoring
# ============================================================
days, ari_scores, latest_ari = monitor_ari(
    df,
    k=k,
    window_days=window_days,
    step_days=step_days,
)

# ============================================================
# Plotly chart
# ============================================================
fig = px.line(
    x=days,
    y=ari_scores,
    labels={"x": "Jours", "y": "ARI"},
    title="Stabilité temporelle des clusters (ARI)",
)

fig.add_hline(y=0.30, line_dash="dash", line_color="orange", annotation_text="Seuil surveillance (0.30)")
fig.add_hline(y=0.20, line_dash="dash", line_color="red", annotation_text="Seuil critique (0.20)")

st.plotly_chart(fig, use_container_width=True)

# ============================================================
# Alerting
# ============================================================
st.subheader("🚨 État du modèle")

if latest_ari >= 0.30:
    st.success(f"ARI = {latest_ari:.3f} → Modèle stable ✅")
elif latest_ari >= 0.20:
    st.warning(f"ARI = {latest_ari:.3f} → Dérive détectée ⚠️")
else:
    st.error(f"ARI = {latest_ari:.3f} → Retrain immédiat requis")

