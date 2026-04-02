# -ML-Segmentation-des-produits-avec-l-lasticit-prix

périmètre : internationnal
# Objectif : 
    -   Suivre le comportement des consommateurs dans me cadre d'une optimisation de stratégie prix :
        -   Prévision de la demande 
        -   Suivre l'évolution de l'élasticité prix des consommateures sur les différentes cat produit 
        

# Etapes :
     
    - Réalisation d'un EDA Afin de capter les relations entre prix, quantité vendus et catégorie de produit
    - Préparation des features en appliquant les principes de l'élastcité prix 
    - Test de plusieurs modèle de machine learning pour réaliser un clustering des catégories produits selon l'élasticité prix 
    - Test de plusieurs modèles poour prévoirs l'élasticité 
    - Déploiement des modèles 
    - Simulation de stratégie prix
  
venv\Scripts\Activate
streamlit run Visualisation/Accueil.py
uvicorn app.main:app --reload