source venv/bin/activate
venv\Scripts\Activate
streamlit run Visualisation/Accueil.py
uvicorn app.main:app --reload