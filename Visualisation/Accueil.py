import streamlit as st
from pathlib import Path

st.title("Présentation du projet et cahier des charges")
st.title("Le projet résumé en slide")

st.set_page_config(layout="wide")


# --- Load slides ---
slides_dir = Path("Visualisation\slide")
slides = sorted([p for p in slides_dir.iterdir()
                 if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp"]])

# --- State ---
if "slide_idx" not in st.session_state:
    st.session_state.slide_idx = 0

n = len(slides)

# --- Controls row ---
c1, c2, c3, c4 = st.columns([1.2, 2.5, 1.2, 2.5])

with c1:
    if st.button("⬅️ Précédent", use_container_width=True):
        st.session_state.slide_idx = (st.session_state.slide_idx - 1) % n

with c2:
    st.markdown(
        f"<div style='text-align:center; font-weight:700; font-size:16px;'>"
        f"Slide {st.session_state.slide_idx + 1} / {n}"
        f"</div>",
        unsafe_allow_html=True
    )

with c3:
    if st.button("Suivant ➡️", use_container_width=True):
        st.session_state.slide_idx = (st.session_state.slide_idx + 1) % n

with c4:
    # Navigation directe
    new_idx = st.selectbox(
        "Aller à",
        options=list(range(n)),
        format_func=lambda i: f"{i+1} — {slides[i].stem}",
        index=st.session_state.slide_idx,
        label_visibility="collapsed",
    )
    st.session_state.slide_idx = new_idx

# --- Frame style ---
st.markdown(
    """
    <style>
    .slide-frame {
        border: 1px solid rgba(120,120,120,0.35);
        border-radius: 18px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.06);
        background: rgba(255,255,255,0.02);
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Display slide inside a "frame" ---
with st.container():
    st.markdown("<div class='slide-frame'>", unsafe_allow_html=True)

    # centrage horizontal
    col1, col2, col3 = st.columns([1,3,1])

    with col2:
        st.image(
            str(slides[st.session_state.slide_idx]),
            width=700
        )
        st.caption(f"Fichier : {slides[st.session_state.slide_idx].name}")

    st.markdown("</div>", unsafe_allow_html=True)



st.title("Cahier des charges — Suivi du comportement consommateur & optimisation de la stratégie prix")

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


st.markdown("""
<div class="section-title">1) Objectifs</div>
<div class="text-block">
<b>Objectif principal :</b><br>
Mettre à disposition un outil de pilotage permettant :
</div>
<div class="text-block dashlist">
<ul>
  <li>d’anticiper la demande à court terme,</li>
  <li>de suivre la sensibilité des consommateurs au prix (élasticité) par catégorie produit.</li>
</ul>
</div>

<div class="text-block">
<b>Objectifs opérationnels :</b><br><br>
<b>Prévision de la demande</b><br>
</div>
<div class="text-block dashlist">
<ul>
  <li>Produire une prévision de la demande avec un horizon de 3 mois.</li>
  <li>Permettre la comparaison entre réalisé et prévu.</li>
</ul>
</div>

<div class="text-block">
<b>Suivi de l’élasticité prix</b><br>
</div>
<div class="text-block dashlist">
<ul>
  <li>Estimer et visualiser l’évolution de l’élasticité prix dans le temps.</li>
  <li>Décliner l’analyse par catégorie produit (et éventuellement par sous-catégorie si disponible).</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">2) Livrables attendus</div>
<div class="text-block">
Un dashboard contenant au minimum 2 graphiques :
</div>
<div class="text-block dashlist">
<ul>
  <li><b>Graphique 1 — Prévision de la demande</b>
    <ul>
      <li>Série temporelle : demande historique + prévision sur 3 mois.</li>
      <li>Affichage avec intervalle de confiance (si modèle le permet).</li>
    </ul>
  </li>
  <li><b>Graphique 2 — Évolution de l’élasticité prix</b>
    <ul>
      <li>Courbe d’élasticité dans le temps.</li>
      <li>Filtre / segmentation par catégorie produit.</li>
    </ul>
  </li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">3) Contraintes métier</div>
<div class="text-block dashlist">
<ul>
  <li>Horizon de prévision obligatoire : 3 mois</li>
  <li>Les visualisations et calculs doivent se mettre à jour chaque mois à partir des nouvelles données disponibles.</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-title">4) Critères d’acceptation</div>
<div class="text-block">
Le livrable est validé si :
</div>
<div class="text-block dashlist">
<ul>
  <li>la prévision affiche bien un horizon 3 mois</li>
  <li>les deux graphiques sont présents et lisibles</li>
  <li>le dashboard se met à jour mensuellement avec les nouvelles données</li>
  <li>l’élasticité est consultable par catégorie produit</li>
  <li>les résultats sont cohérents (pas de valeurs aberrantes non expliquées, courbes exploitables)</li>
</ul>
</div>
""", unsafe_allow_html=True)
