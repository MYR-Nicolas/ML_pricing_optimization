import streamlit as st
from PIL import Image

st.set_page_config(layout="wide", page_title="EDA - Analyse des ventes")

st.markdown("""
<style>
.main-title {
    font-size: 40px;
    font-weight: 700;
    margin-bottom: 10px;
}
.section-title {
    font-size: 26px;
    font-weight: 600;
    margin-top: 34px;
    margin-bottom: 10px;
}
.text-block {
    font-size: 16px;
    line-height: 1.6;
}
.bullets {
    margin-top: 8px;
}
.bullets ul {
    margin: 0;
    padding-left: 18px;
    list-style-type: none;
}
.bullets li {
    margin: 6px 0;
}
.bullets li::before {
    content: "- ";
}
hr {
    margin-top: 24px;
    margin-bottom: 24px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Analyse exploratoire des ventes</div>', unsafe_allow_html=True)

st.markdown("""
<div class="text-block">
Cette section présente l’analyse exploratoire du dataset transactionnel. L’objectif est de comprendre la structure des quantités,
la dynamique des prix, la saisonnalité, la relation prix-volume et les implications directes pour la modélisation.
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">Insights métier et data</div>
<div class="text-block bullets">
<ul>
<li>Identification de deux segments clients : professionnels vs consommateurs particuliers, avec des comportements d’achat distincts.</li>
<li>La quantité modale est de 1 unité (87 % des transactions), ce qui suggère un comportement B2C dominant et une faible volumétrie par transaction.</li>
<li>Distribution des prix concentrée entre 200 et 1 000, indiquant un positionnement majoritairement milieu de gamme.</li>
<li>Présence de gammes produits différenciées (classique vs premium), notamment sur la catégorie <i>kurta</i>, impliquant une hétérogénéité de structure de prix.</li>
<li>Les catégories générant le plus de volume sont <i>kurta</i>, <i>set</i> et <i>top</i>, constituant le cœur du business et les principales variables d’intérêt pour la modélisation de la demande.</li>
</ul>

<b>Implications pour la modélisation :</b>
<ul>
<li>Absence de relation linéaire marquée entre prix et quantité → pertinence de modèles non linéaires (tree-based, boosting, réseaux de neurones).</li>
<li>Non-normalité des résidus du modèle baseline → violation des hypothèses OLS classiques, confirmant l’intérêt d’approches ML robustes.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">Description du dataset</div>
<div class="text-block bullets">
<ul>
<li>Dataset transactionnel de 37 432 observations et 11 variables.</li>
<li>Empreinte mémoire : 3,1 GB (~6 % de la RAM disponible). Les opérations d’EDA et de feature engineering restent compatibles avec un workflow pandas standard.</li>
<li>Enrichissement du dataset via jointure avec <i>International Sale Report</i> afin d’ajouter les catégories produits.</li>
<li>Présence de colonnes mixtes nécessitant un nettoyage et une normalisation des types.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">Data quality & data cleaning</div>
<div class="text-block bullets">
<ul>
<li>1 371 valeurs manquantes sur la variable SKU (0,038 %), impact négligeable sur la volumétrie globale.</li>
<li>Absence de doublons transactionnels.</li>
<li>Détection d’incohérences structurelles dans les colonnes mixtes :</li>
<ul>
    <li>Des dates dans la colonne <i>customer</i>.</li>
    <li>Des identifiants clients dans la colonne <i>DATE</i> (18 321 occurrences, &gt;50 %).</li>
    <li>Formats incorrects dans la colonne <i>SIZE</i>.</li>
</ul>
<li>Nettoyage réalisé via regex et règles de parsing, suivi d’un contrôle qualité post-transformation.</li>
<li>Conversion explicite des types de données et validation des erreurs de cast.</li>
</ul>

<b>Gestion des outliers :</b>
<ul>
<li>Quantités élevées expliquées par des acheteurs professionnels (hypothèse validée).</li>
<li>Dispersion des prix liée à des gammes produits distinctes.</li>
<li>Décision méthodologique : conservation des outliers afin de préserver l’information économique et la variance utile à la modélisation.</li>
</ul>
</div>
""", unsafe_allow_html=True)

# Chargement des visualisations 

img_dist_qte  = Image.open("graphique/EDA/Distribution_qty.png")
img_dist_prix = Image.open("graphique/EDA/Distribution_price.png")
img_dist_cat  = Image.open("graphique/EDA/Distribution_category.png")
img_corr      = Image.open("graphique/EDA/scatter.png")
img_evol_prix = Image.open("graphique/EDA/trend_price.png")
img_evol_qte  = Image.open("graphique/EDA/trend_qty.png")
img_qq        = Image.open("graphique/EDA/QQ_plot.png")
# Modification de la taille des visualisations 

def taille_img(img):
    w, h = img.size
    st.image(img, width=int(w/1.5))

st.markdown('<div class="section-title">1. Distribution des quantités</div>', unsafe_allow_html=True)
taille_img(img_dist_qte)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>87 % des transactions à 1 unité → distribution fortement asymétrique.</li>
<li>Longue traîne associée à des achats professionnels (B2B).</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">2. Distribution des prix</div>', unsafe_allow_html=True)
taille_img(img_dist_prix)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>Distribution concentrée entre 200 et 1 000, positionnement majoritairement milieu de gamme.</li>
<li>Présence de produits premium, cohérente avec une segmentation des gammes.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">3. Répartition des produits par catégorie</div>', unsafe_allow_html=True)
taille_img(img_dist_cat)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>Les catégories <i>kurta</i>, <i>set</i> et <i>top</i> génèrent le plus de volume.</li>
<li>Ces catégories représentent le cœur business et doivent être prioritaires dans l’analyse et la modélisation.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">4. Relation prix – quantité</div>', unsafe_allow_html=True)
taille_img(img_corr)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>Corrélation prix-quantité globalement faible voire nulle selon les segments.</li>
<li>Absence de relation linéaire globale entre prix et volume → approche non paramétrique recommandée.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">5. Évolution temporelle des prix</div>', unsafe_allow_html=True)
taille_img(img_evol_prix)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>Stabilisation des prix par catégorie : Set (800–1 100), Kurta (400–600), Top (300–550).</li>
<li>Anomalie visible sur <i>kurta</i> (mai 2022) pouvant refléter une incohérence ou une erreur de pricing.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">6. Évolution des quantités</div>', unsafe_allow_html=True)
taille_img(img_evol_qte)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>Pics de ventes en fin d’année (effets calendaires et périodes promotionnelles type Black Friday).</li>
<li>Intensification notable des ventes en 2023, indiquant une saisonnalité plus marquée.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-title">7. Diagnostic statistique (baseline OLS)</div>', unsafe_allow_html=True)
taille_img(img_qq)
st.markdown("""
<div class="text-block bullets">
<ul>
<li>R² ≈ 0,22 : pouvoir explicatif modéré.</li>
<li>Multicolinéarité élevée due au grand nombre de variables catégorielles.</li>
<li>Résidus non gaussiens (asymétrie, queues épaisses, outliers) → bruit élevé et cible non gaussienne.</li>
<li>L’OLS sert de modèle exploratoire pour détecter du signal linéaire et orienter le feature engineering.</li>
<li>Les résultats constituent un benchmark avant l’entraînement de modèles ML plus flexibles.</li>
</ul>
</div>
""", unsafe_allow_html=True)



