# Segmentation Client Avancée - RFM, K-Means & DBSCAN
[Code source GitHub | Démo interactive](https://customer-segmentation-ecommerce-rfm-kmeans-dbscan-ogdvmz6rtytd.streamlit.app/)

### Démo :
<img src="images/photo/RFM.png" width="400">

### CONTEXTE BUSINESS
**Problématique**
Une entreprise e-commerce dispose de milliers de clients, mais :
- elle cherche à savoir qui prioriser
- ne distingue pas les clients rentables
- ignore les clients atypiques ou à risque
  
**Objectif**

Construire une segmentation client exploitable métier pour :
- augmenter la valeur client
- réduire le churn
- guider les actions marketing & CRM
  
**DONNÉES & PRÉPARATION**

- Dataset réel Olist (Brésil)
- +100 000 transactions
- Niveau transactionnel → client
  
**Nettoyage & préparation**
- Filtrage des commandes livrées
- Suppression des doublons
- Conversion des dates
- Agrégation client (RFM)
  
**Résultat obtenu :** une base client propre et actionnable
## Méthodes
### Méthode 1 - SEGMENTATION RFM
- Recency : récence du dernier achat
- Frequency : nombre de commandes
- Monetary : montant total dépensé
- Scoring 1–5 (quantiles + fallback)

**Segments obtenus**
- Champions
- Clients fidèles
- Clients prometteurs
- Nouveaux clients
- À surveiller
- À réactiver
- À risque
- Perdus

**Impact business**
- Priorisation marketing
- Lecture simple pour équipes métier
- Vision valeur client claire
### Méthode 2 - K-MEANS (SEGMENTATION COMPORTEMENTALE)
**Objectif**
Identifier des groupes de clients similaires, sans règles métiers.

**Variables utilisées**

- Recency
- Frequency
- Monetary
- Score moyen d’avis
- Paiements moyens

**Démarches**
- Normalisation MinMax
- Sélection de K via Silhouette Score
- Visualisations interactives (Plotly)
**Résultats obtenus**
- Meilleur K identifié automatiquement
- Clusters interprétés par profil moyen
- Génération automatique d’interprétations business
### Méthode 3 - DBSCAN (CLIENTS ATYPIQUES)
**Objectif**

Détecter les clients hors normes :
- comportements extrêmes
- anomalies
- profils à fort risque ou opportunité

**Démarches**

- Standardisation robuste
- DBSCAN (eps + min_samples)
- K-distance plot automatique

**Résultat obtenu**
- Label -1 = clients atypiques
- Export vers fichier actionnable

### VUE STRATÉGIQUE FINALE (CEO / CRM VIEW)
Fusion **RFM + K-Means + DBSCAN**

**Typologie clients**

- VIP
- À Risque
- Standard
- Atypiques
- Vue orientée décision & priorisation, pas algorithme.
  
**DASHBOARD STREAMLIT**

Fonctionnalités :

- Upload CSV
- Filtres dynamiques
- Visualisations interactives
- Interprétation automatique
- Export clients actionnables
- déploiement cloud Streamlit
## Monitoring – Stabilité de la Segmentation
[Code source GitHub | Démo interactive](https://customer-segmentation-ecommerce-rfm-kmeans-dbscan-kje2pukjdu2t.streamlit.app/)
### Démo :
<img src="images/photo/ARI.png" width="400">

**Objectif du monitoring**

Dans un contexte de segmentation **non supervisée (K-Means / DBSCAN)**, il n’existe pas de "vérité terrain".

Le monitoring vise donc à :
- Détecter les dérives comportementales clients
- Garantir la stabilité des clusters dans le temps
- Décider quand retrainer le modèle
  
**Principe retenu : Adjusted Rand Index (ARI)**
  
L’**Adjusted Rand Index (ARI)** mesure la similarité entre deux partitions de clusters :

- ARI = 1 → clusters parfaitement stables
- ARI ≈ 0 → structure aléatoire
- ARI < 0 → dérive majeure
Ce **monitoring temporel (Rolling Window)**
- Fenêtre historique (window_days) : Période de référence servant de baseline.
- Pas temporel (step_days) : Fréquence de recalcul de la segmentation

**Exemple**
- Fenêtre = 182 jours
- Pas = 7 jours

Le modèle est comparé chaque semaine à la segmentation de référence (année N-1).

**Politique d’alertes**
**ARI  	              Statut     	              Action                      Décision**
ARI ≥ 0.30	         Stable	                    Aucune action               Retrain trimestriel
0.20 ≤ ARI < 0.30	   Dérive détectée	          Surveillance accrue         Retrain mensuel
ARI < 0.20	         Dérive critique	          Retrain immédiat            Retrain immédiat

Cette logique est implémentée directement dans le pipeline de monitoring.

**Visualisation & Dashboard**

Le monitoring est exposé via un dashboard Streamlit interactif :

- Courbe ARI dans le temps
- Seuils métiers visuels
- Alertes en temps réel
- Paramétrage dynamique (k, fenêtre, pas)

**Valeur business**

- Anticipation de la perte de performance
- Meilleure priorisation clients (VIP 
