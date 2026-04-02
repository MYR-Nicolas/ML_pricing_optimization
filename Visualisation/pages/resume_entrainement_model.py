import streamlit as st
from PIL import Image

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


.dashlist ul{
    list-style: none ;
    margin: 0 ;
    padding-left: 0 ;
}
.dashlist ul ul{
    margin-left: 26px ;
    margin-top: 6px ;
}
.dashlist li{
    margin: 6px 0 ;
}
.dashlist li::before{
    content: "- " ;
}
</style>
""", unsafe_allow_html=True)

# TEXTE (Objectifs / Métriques / Modèles / Processus) SANS AUCUNE PUCE RONDE/CARRÉE
st.markdown("""
<div class="section-title">Objectifs</div>
<div class="text-block dashlist">
<ul>
  <li>Construire un modèle de prévision des ventes (quantités) à partir des variables transactionnelles, catégorielles et temporelles.</li>
  <li>Comparer plusieurs familles de modèles (linéaires vs non linéaires) afin d’identifier le meilleur compromis performance / robustesse / coût computationnel.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">Choix des métriques</div>
<div class="text-block dashlist">
<ul>
  <li><b>MAE (métrique principale)</b>
    <ul>
      <li>Cible asymétrique avec présence d’outliers liés à deux profils d’acheteurs.</li>
      <li>Hypothèse d’un coût d’erreur approximativement linéaire.</li>
      <li>Métrique robuste aux valeurs extrêmes.</li>
    </ul>
  </li>

  <li><b>RMSE (métrique complémentaire)</b>
    <ul>
      <li>Pénalisation plus forte des grandes erreurs.</li>
      <li>Permet d’évaluer le risque opérationnel lié aux fortes déviations.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">Choix des modèles</div>
<div class="text-block dashlist">
<ul>
  <li><b>OLS (outil de diagnostic structurel)</b>
    <ul>
      <li>Analyse des coefficients et des résidus.</li>
      <li>Détection de :
        <ul>
          <li>non-linéarité</li>
          <li>multicolinéarité</li>
          <li>interactions complexes</li>
        </ul>
      </li>
      <li>Rôle : analyse exploratoire et compréhension du signal linéaire.</li>
    </ul>
  </li>

  <li><b>GLM (modèle de référence)</b>
    <ul>
      <li>Baseline robuste et interprétable.</li>
      <li>Plus stable que l’OLS dans certains contextes.</li>
      <li>Rôle : benchmark pour les modèles plus complexes.</li>
    </ul>
  </li>

  <li><b>SGDRegressor (linéaire régularisé)</b>
    <ul>
      <li>Contexte : variables catégorielles haute dimension (One-Hot Encoding).</li>
      <li>Apports :
        <ul>
          <li>régularisation (L1/L2)</li>
          <li>scalabilité</li>
          <li>meilleure généralisation</li>
        </ul>
      </li>
      <li>Rôle : mesurer les limites d’un modèle linéaire régularisé.</li>
    </ul>
  </li>

  <li><b>Random Forest (modèle non paramétrique)</b>
    <ul>
      <li>Contexte :
        <ul>
          <li>non-linéarité détectée à l’EDA</li>
          <li>interactions entre variables</li>
        </ul>
      </li>
      <li>Apports :
        <ul>
          <li>capture des interactions</li>
          <li>robustesse aux outliers</li>
          <li>absence d’hypothèse paramétrique forte</li>
        </ul>
      </li>
      <li>Limites :
        <ul>
          <li>variance plus élevée</li>
          <li>coût computationnel supérieur</li>
        </ul>
      </li>
    </ul>
  </li>

  <li><b>HistGradientBoostingRegressor (boosting)</b>
    <ul>
      <li>Objectif :
        <ul>
          <li>réduire le biais des modèles linéaires</li>
          <li>réduire la variance observée sur Random Forest</li>
          <li>optimiser la performance tabulaire</li>
        </ul>
      </li>
      <li>Rôle : candidat final (compromis biais / variance).</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">Processus de modélisation</div>
<div class="text-block dashlist">
<ul>
  <li><b>Pré-processing</b>
    <ul>
      <li>Split chronologique des datasets.</li>
      <li>Encodage spécifique pour le modèle baseline.</li>
      <li>Construction d’une Pipeline (ColumnTransformer + encodage + modèle).</li>
    </ul>
  </li>

  <li><b>Étape 1 : Baseline</b>
    <ul>
      <li>Entraînement du modèle de référence.</li>
      <li>Évaluation MAE / RMSE.</li>
    </ul>
  </li>

  <li><b>Étape 2 : Sélection du meilleur dataset</b>
    <ul>
      <li>Premier entraînement comparatif :
        <ul>
          <li>SGDRegressor</li>
          <li>Random Forest</li>
          <li>HistGradientBoostingRegressor</li>
        </ul>
      </li>
      <li>Comparaison des métriques MAE et RMSE.</li>
      <li>Vérification que les catégories principales sont représentées dans chaque split.</li>
    </ul>
  </li>

  <li><b>Étape 3 : Sélection du meilleur modèle</b>
    <ul>
      <li>Élargissement des hyperparamètres.</li>
      <li>Comparaison des métriques.</li>
      <li>Analyse des ressources utilisées (temps d’entraînement, consommation mémoire).</li>
    </ul>
  </li>

  <li><b>Étape 4 : Optimisation finale</b>
    <ul>
      <li>GridSearch élargi sur le meilleur modèle.</li>
      <li>Analyse de la courbe d’apprentissage (diagnostic biais / variance).</li>
      <li>Analyse détaillée des métriques.</li>
      <li>Interprétation via SHAP (importance globale et contributions locales).</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =========================
# PARTIE RESULTATS
# =========================

img_dataset  = Image.open("Notebook/graphique/entrainement/dataset.png")
img_eval_model = Image.open("Notebook/graphique/entrainement/eval_model.png")
img_learning_curve  = Image.open("Notebook/graphique/entrainement/learning_curve.png")
img_split_df1 = Image.open("Notebook/graphique/entrainement/split_df1.png")
img_split_df2 = Image.open("Notebook/graphique/entrainement/split_df2.png")

def taille_img(img):
    w, h = img.size
    st.image(img, width=int(w/1.5))

# =====================================================
# 1. COMPARAISON DES DATASETS
# =====================================================

st.markdown('<div class="section-title">Comparaison des datasets</div>', unsafe_allow_html=True)
taille_img(img_dataset)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li>Le dataset le plus performant est <b>df_feature_1</b>, bien qu’il contienne moins de variables explicatives.
    <ul>
      <li>Volume de données nettement plus important (≈ 7x plus d’observations).</li>
      <li>Amélioration de la capacité de généralisation.</li>
      <li>Réduction du risque de sur-apprentissage.</li>
    </ul>
  </li>

  <li>En validation croisée comme sur le jeu de test, les métriques sont significativement meilleures.
    <ul>
      <li>MAE et RMSE plus faibles.</li>
      <li>La quantité de données a un impact plus déterminant que le nombre de features.</li>
      <li>La profondeur statistique prime sur la complexité du feature engineering.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 2. QUALITE DU SPLIT
# =====================================================

st.markdown('<div class="section-title">Qualité du split des données</div>', unsafe_allow_html=True)
taille_img(img_split_df1)
taille_img(img_split_df2)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li>Le dataset 1 présente un découpage moins homogène.
    <ul>
      <li>Distribution déséquilibrée entre train, validation et test.</li>
      <li>Variance plus élevée dans les métriques.</li>
    </ul>
  </li>

  <li>Le second dataset montre un split plus équilibré.
    <ul>
      <li>Meilleure homogénéité des distributions.</li>
      <li>Métriques plus stables.</li>
      <li>Évaluation plus fiable.</li>
    </ul>
  </li>

  <li>Un split homogène améliore la robustesse de l’évaluation.
    <ul>
      <li>Réduction du biais d’estimation.</li>
      <li>Comparabilité accrue entre modèles.</li>
      <li>Sélection de modèle plus fiable.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 3. SELECTION DATASET
# =====================================================

st.markdown('<div class="section-title">Sélection du dataset</div>', unsafe_allow_html=True)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li>Le dataset <b>df_feature_1</b> est retenu.
    <ul>
      <li>Meilleures performances globales.</li>
      <li>Résultats supérieurs en CV et test.</li>
      <li>Meilleure généralisation hors échantillon.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 4. COMPARAISON MODELES
# =====================================================

st.markdown('<div class="section-title">Comparaison des modèles</div>', unsafe_allow_html=True)
taille_img(img_eval_model)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li>Cohérence globale entre validation croisée et test.
    <ul>
      <li>Bonne capacité de généralisation.</li>
      <li>Absence de sur-apprentissage majeur.</li>
    </ul>
  </li>

  <li><b>RandomForestRegressor</b> : meilleures performances.
    <ul>
      <li>MAE et RMSE les plus faibles.</li>
      <li>Écart CV / test limité.</li>
      <li>Bon compromis biais / variance.</li>
    </ul>
  </li>

  <li><b>SGDRegressor</b> : performances intermédiaires.
    <ul>
      <li>Sensibilité plus forte aux variations.</li>
      <li>Pas de sur-apprentissage marqué.</li>
    </ul>
  </li>

  <li><b>HistGradientBoostingRegressor</b> : erreurs plus élevées.
    <ul>
      <li>Généralisation correcte.</li>
      <li>Performance globale inférieure sur ce dataset.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 5. COUT SYSTEME
# =====================================================

st.markdown('<div class="section-title">Analyse des performances système</div>', unsafe_allow_html=True)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li><b>SGDRegressor</b> : modèle le plus rapide.
    <ul>
      <li>Latence très faible (~13 ms).</li>
      <li>Débit très élevé.</li>
      <li>Taille modèle < 1MB.</li>
    </ul>
  </li>

  <li><b>RandomForestRegressor</b> : modèle le plus coûteux.
    <ul>
      <li>Latence élevée.</li>
      <li>Taille modèle ~120MB.</li>
      <li>Temps de chargement important.</li>
      <li>Mais meilleure performance prédictive.</li>
    </ul>
  </li>

  <li><b>HistGradientBoostingRegressor</b> : compromis.
    <ul>
      <li>Latence faible.</li>
      <li>Taille modèle très faible.</li>
      <li>CPU plus élevé.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 6. MODELE FINAL
# =====================================================

st.markdown('<div class="section-title">Choix du modèle final</div>', unsafe_allow_html=True)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li><b>RandomForestRegressor</b> est retenu.
    <ul>
      <li>Meilleure performance prédictive.</li>
      <li>Stabilité CV / test.</li>
      <li>Bon compromis biais / variance.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

# =====================================================
# 7. LEARNING CURVE
# =====================================================

st.markdown('<div class="section-title">Analyse de la learning curve</div>', unsafe_allow_html=True)
taille_img(img_learning_curve)

st.markdown("""
<div class="text-block dashlist">
<ul>
  <li>Avec peu de données : sur-apprentissage marqué.
    <ul>
      <li>Écart train / validation élevé.</li>
      <li>Variance importante.</li>
    </ul>
  </li>

  <li>Augmentation des données → meilleure généralisation.
    <ul>
      <li>Réduction de l’écart train / validation.</li>
      <li>Stabilisation vers 5 000 observations.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)