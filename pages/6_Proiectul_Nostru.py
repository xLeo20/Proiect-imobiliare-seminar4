import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import plotly.express as px

st.title("Proiect de Grup — Predicții Imobiliare")
st.markdown("Această pagină antrenează un model Machine Learning pentru a prezice prețurile.")
st.markdown("---")

# ── 1. Datele ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Poți înlocui "imobiliare_romania.csv" cu dataset-ul tău dacă ai altul
    return pd.read_csv("imobiliare_romania.csv")

df = load_data()

# ── 2. Preprocesare interactivă ──────────────────────────────────────
st.subheader("1. Configurează preprocesarea datelor")
st.markdown("Alege cum vrei să tratăm proprietățile care nu au anul de construcție completat:")

# Cerința 1: Widget interactiv pentru preprocesare
metoda_lipsa = st.radio(
    "Metoda de imputare pentru 'An construcție':",
    ["Înlocuiește cu media", "Înlocuiește cu mediana", "Elimină rândurile complet"],
    horizontal=True
)

# Aplicăm logica de preprocesare aleasă
df_ml = df.copy()

if metoda_lipsa == "Înlocuiește cu media":
    df_ml["An construcție"] = df_ml["An construcție"].fillna(df_ml["An construcție"].mean())
elif metoda_lipsa == "Înlocuiește cu mediana":
    df_ml["An construcție"] = df_ml["An construcție"].fillna(df_ml["An construcție"].median())
else:
    df_ml = df_ml.dropna(subset=["An construcție"])

# Pentru casele fără etaj, punem etajul 0 (parter)
df_ml["Etaj"] = df_ml["Etaj"].fillna(0)

# Păstrăm doar coloanele numerice pentru a păstra modelul simplu și rapid
df_ml = df_ml.select_dtypes(include=["number"]).dropna()

# ── 3. Configurare și antrenare model ───────────────────────────────
st.markdown("---")
st.subheader("2. Configurează modelul Machine Learning")

# Cerința 2: st.form și st.spinner
with st.form("config_ml"):
    col1, col2 = st.columns(2)
    with col1:
        target = st.selectbox("Ce dorim să prezicem? (Variabila țintă)", ["Preț (EUR)", "Preț/mp (EUR)"])
    with col2:
        test_size = st.slider("Proporție date pentru testare (Test size)", 0.1, 0.4, 0.2, 0.05)
    
    submit = st.form_submit_button("Antrenează modelul")

if submit:
    # Afișăm spinner-ul cât timp modelul calculează
    with st.spinner("Modelul învață din date... Te rugăm să aștepți."):
        
        # Pregătim datele X (caracteristici) și y (ținta)
        X = df_ml.drop(columns=[target])
        y = df_ml[target]
        
        # Împărțim datele
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Antrenăm modelul (Regresie pentru a prezice prețul exact)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_tr, y_tr)
        
        # Facem predicții pe datele de test
        y_pred = model.predict(X_te)

    st.success("Antrenare completă cu succes!")

    # ── 4. Rezultate vizibile ──────────────────────────────────────
    st.markdown("---")
    st.subheader("3. Rezultatele Modelului")
    
    # Calculăm metricile
    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))

    # Cerința 3: st.metric
    m1, m2 = st.columns(2)
    m1.metric("R² (Precizia modelului)", f"{r2:.3f}")
    m2.metric("RMSE (Eroarea medie)", f"{rmse:,.0f} €")

    # Cerința 3: Grafic vizual
    st.markdown("**Comparație: Prețul Real vs. Prețul Prezis de model**")
    fig = px.scatter(
        x=y_te, y=y_pred,
        labels={"x": "Preț Real", "y": "Preț Prezis"},
        color_discrete_sequence=["#2d6a4f"],
        opacity=0.7
    )
    
    # Adăugăm linia diagonală (linia de predicție perfectă)
    max_val = max(y_te.max(), y_pred.max())
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="red", dash="dash")
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Punctele care se află pe linia roșie întreruptă reprezintă predicții perfecte!")