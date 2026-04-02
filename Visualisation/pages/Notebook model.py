import streamlit as st

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

st.title("Notebook entrainement des modèles")

st.markdown("""
<div class="section-title">Structures du Notebook</div>

<div class="text-block">
<div class="dashlist">
<ul>
  <li>Importation</li>
  <li>Pré-processing</li>
  <li>Outils d’évaluation des modèles
    <ul>
      <li>Analyse des métriques</li>
      <li>Analyse des ressources utilisées</li>
      <li>Courbe d’apprentissage</li>
      <li>Interprétation SHAP</li>
    </ul>
  </li>
  <li>Entraînement
    <ul>
      <li>Baseline</li>
      <li>ML</li>
    </ul>
  </li>
</ul>
</div>
</div>
""", unsafe_allow_html=True)

with open("Notebook/test_model.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)