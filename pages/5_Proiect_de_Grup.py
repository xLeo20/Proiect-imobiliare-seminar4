import streamlit as st

st.title("Extinderea Dashboard-ului")
st.markdown("**Opțional · Pentru cei care vor să meargă mai departe**")
st.markdown("---")

st.markdown("""
### Contexul

În seminarul anterior ați construit un dashboard interactiv pe un dataset ales de voi.
Acum știti suficient de mult despre preprocesare și Machine Learning ca să îl faceți
să și **gândească** — nu doar să afișeze date, ci să facă predicții.

Provocarea este simplă: **adăugați o pagină nouă** în dashboard-ul vostru existent.
""")

st.info("""
Aceasta nu este o temă obligatorie.
Este pentru cei care vor să vadă cum arată un dashboard complet —
de la date brute la predicții interactive.
""")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# CE TREBUIE SĂ FACĂ
# ═════════════════════════════════════════════════════════════════
st.header("Ce adăugați")

st.markdown("""
Creați un fișier nou în folderul `pages/` al proiectului —
de exemplu `3_Machine_Learning.py` sau `4_Predictii.py`.

Pagina trebuie să conțină **trei lucruri**:
""")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **1. Preprocesare interactivă**

    Cel puțin o decizie lăsată utilizatorului:
    metoda de imputare, tratarea outlierilor
    sau encoding-ul unei coloane categorice.

    Folosiți `st.radio` sau `st.selectbox`
    pentru a oferi opțiunile.
    """)

with col2:
    st.markdown("""
    **2. Antrenare model**

    Un model simplu antrenat pe datele voastre —
    regresie dacă vreți să preziceți o valoare numerică,
    clasificare dacă vreți să încadrați în categorii.

    Folosiți `st.form` pentru configurare
    și `st.spinner` în timpul antrenării.
    """)

with col3:
    st.markdown("""
    **3. Rezultate vizibile**

    Cel puțin o metrică de performanță
    afișată cu `st.metric` și un grafic
    care să arate cât de bine funcționează modelul.

    Un grafic Real vs. Prezis sau o confusion matrix
    spun mai mult decât un număr singur.
    """)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# TEMPLATE
# ═════════════════════════════════════════════════════════════════
st.header("Template de pornire")

st.markdown("""
Copiați codul de mai jos într-un fișier nou în folderul `pages/` al proiectului 
și adaptați-l pentru dataset-ul vostru.
Liniile marcate cu `# TODO` sunt cele pe care trebuie să le modificați.
""")

st.code("""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
import plotly.express as px

st.title("Predicții — Machine Learning")

# ── Date ──────────────────────────────────────────────────────────
# TODO: înlocuiți cu modul în care încărcați datele în proiectul vostru
# Dacă folosiți session_state din pagina principală:
# df = st.session_state["df"]
# Dacă încărcați direct:
# df = pd.read_csv("numele_fisierului.csv")

# ── Preprocesare interactivă ──────────────────────────────────────
st.subheader("Configurează preprocesarea")

metoda = st.radio(
    "Cum tratăm valorile lipsă?",
    ["Înlocuiește cu media", "Înlocuiește cu mediana", "Elimină rândurile"],
    horizontal=True
)

# TODO: aplicați metoda aleasă pe coloanele cu valori lipsă

# ── Configurare și antrenare model ───────────────────────────────
st.subheader("Configurează modelul")

with st.form("config_ml"):
    col1, col2 = st.columns(2)
    with col1:
        # TODO: înlocuiți cu coloana țintă din dataset-ul vostru
        target = st.selectbox("Variabilă țintă", ["coloana_voastra"])
    with col2:
        test_size = st.slider("Proporție date de test", 0.1, 0.4, 0.2)
    submit = st.form_submit_button("Antrenează modelul")

if submit:
    with st.spinner("Antrenez modelul..."):
        # TODO: pregătiți X și y din dataset-ul vostru
        # X = df.drop(columns=[target]).select_dtypes("number")
        # y = df[target]
        # X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size)
        # model = RandomForestRegressor()  # sau Classifier
        # model.fit(X_tr, y_tr)
        # y_pred = model.predict(X_te)
        pass

    st.success("Antrenare completă!")

    # TODO: afișați metricile relevante
    # col1, col2 = st.columns(2)
    # col1.metric("R²", f"{r2_score(y_te, y_pred):.3f}")
    # col2.metric("RMSE", f"{mean_squared_error(y_te, y_pred, squared=False):,.0f}")

    # TODO: adăugați un grafic al rezultatelor
""", language="python")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# SFATURI
# ═════════════════════════════════════════════════════════════════
st.header("Câteva sfaturi practice")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    **Dacă nu știți ce coloană să preziceți**

    Gândiți-vă la întrebarea cea mai naturală pentru dataset-ul vostru.
    Pentru mașini second-hand: *Care este prețul?*
    Pentru studenti: *Ce notă va lua?*
    Pentru filme: *Ce rating va primi?*

    Dacă aveți o coloană numerică continuă → **regresie**.
    Dacă aveți o coloană cu categorii → **clasificare**.
    """)

with col2:
    st.markdown("""
    **Dacă modelul dă rezultate slabe**

    Un R² mic sau o Accuracy slabă nu înseamnă că ați greșit ceva.
    Înseamnă că datele sunt greu de prezis cu variabilele disponibile
    — și asta e o concluzie validă.

    Încercați să schimbați metoda de preprocesare sau să eliminați
    coloane care nu au legătură logică cu variabila țintă.
    """)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════
# PREZENTARE OPȚIONALĂ
# ═════════════════════════════════════════════════════════════════
st.markdown("""
### Vreți să prezentați?

Dacă sunteți mulțumiți de rezultat, sunteți bineveniți să prezentați **2 minute**
la începutul seminarului următor — ce dataset ați folosit, ce model ați antrenat
și ce ați descoperit.

Nu este obligatoriu.
E pur și simplu o ocazie să arătați ce ați construit :).
Succes!
""")

st.markdown("---")
st.markdown("""
<div style="text-align:center; padding: 8px 0 4px 0;">
    <div style="font-size: 20px; font-weight: 700; color: #2d6a4f;">Mult succes!</div>
    <div style="font-size: 12px; color: #999; margin-top: 4px;">
        Seminar 4 · Python & Streamlit & Machine Learning
    </div>
</div>
""", unsafe_allow_html=True)