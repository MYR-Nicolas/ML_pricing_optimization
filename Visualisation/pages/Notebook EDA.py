import streamlit as st

# Gestion du style
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

st.title("Notebook analyse exploratoire des données")

st.markdown("""
<div class="section-title">Structure du Notebook</div>
<div class="text-block dashlist">
<ul>
  <li>Importation</li>
  <li>Chargement et informations générales des datasets</li>
  <li>Data cleaning</li>
  <li>Analyse</li>
  <li>Data featuring</li>
  <li>Vérification Final</li>
</ul>
</div>
""", unsafe_allow_html=True)

with open("Notebook/exploratory_data_analysis.html", "r", encoding="utf-8") as f:
    html = f.read()

st.components.v1.html(html, height=800, scrolling=True)