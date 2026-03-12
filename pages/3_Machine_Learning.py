import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, classification_report, confusion_matrix
)
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
import time

# ── Helpers ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("imobiliare_romania.csv")

def preproceseaza(df_raw, cfg):
    """Reproduce pipeline-ul de pe Pagina 2 cu o configurație dată."""
    df = df_raw.copy()

    # Pasul 1 — valori lipsă
    if cfg["metoda_an"] == "Mean":
        imp = SimpleImputer(strategy="mean")
        df["An construcție"] = imp.fit_transform(df[["An construcție"]]).ravel().round().astype(float)
    elif cfg["metoda_an"] == "Median":
        imp = SimpleImputer(strategy="median")
        df["An construcție"] = imp.fit_transform(df[["An construcție"]]).ravel().round().astype(float)
    elif cfg["metoda_an"] == "Forward Fill (ffill)":
        df["An construcție"] = df["An construcție"].ffill().round().astype(float)
    elif cfg["metoda_an"] == "Backward Fill (bfill)":
        df["An construcție"] = df["An construcție"].bfill().round().astype(float)
    elif cfg["metoda_an"] == "KNN Imputer":
        knn = KNNImputer(n_neighbors=5)
        cols = ["An construcție", "Suprafață (mp)", "Preț (EUR)"]
        df[cols] = knn.fit_transform(df[cols])
        df["An construcție"] = df["An construcție"].round().astype(float)

    if cfg["metoda_etaj"] == "Completează cu 0 (parter)":
        df["Etaj"] = df["Etaj"].fillna(0).astype(float)
    elif cfg["metoda_etaj"] == "Median":
        df["Etaj"] = df["Etaj"].fillna(df["Etaj"].median()).astype(float)
    elif cfg["metoda_etaj"] == "KNN Imputer":
        knn2 = KNNImputer(n_neighbors=5)
        df[["Etaj","Suprafață (mp)","Nr. camere"]] = knn2.fit_transform(
            df[["Etaj","Suprafață (mp)","Nr. camere"]]
        )
        df["Etaj"] = df["Etaj"].round().astype(float)
    # "Lasă lipsă" — rămâne cum e

    # Pasul 2 — outlieri
    if cfg["metoda_outlieri"] == "Elimină rândurile outlieri":
        Q1 = df["Preț (EUR)"].quantile(0.25)
        Q3 = df["Preț (EUR)"].quantile(0.75)
        IQR = Q3 - Q1
        df = df[
            (df["Preț (EUR)"] >= Q1 - 1.5 * IQR) &
            (df["Preț (EUR)"] <= Q3 + 1.5 * IQR)
        ].reset_index(drop=True)
    elif cfg["metoda_outlieri"] == "Capping la percentile":
        low  = df["Preț (EUR)"].quantile(0.01)
        high = df["Preț (EUR)"].quantile(0.99)
        df["Preț (EUR)"] = df["Preț (EUR)"].clip(low, high)

    # Pasul 3 — encoding
    if cfg["enc_tip"] == "Label Encoding (0/1)":
        df["Tip"] = (df["Tip"] == "Apartament").astype(int)
    else:
        df = pd.get_dummies(df, columns=["Tip"], prefix="Tip")

    if cfg["enc_oras"] == "Label Encoding":
        le = LabelEncoder()
        df["Oraș"] = le.fit_transform(df["Oraș"])
    else:
        df = pd.get_dummies(df, columns=["Oraș"], prefix="Oraș")

    if cfg.get("cartier_inclus", True):
        if cfg.get("enc_cartier", "Label Encoding") == "Label Encoding":
            le2 = LabelEncoder()
            df["Cartier"] = le2.fit_transform(df["Cartier"].astype(str))
        else:
            df = pd.get_dummies(df, columns=["Cartier"], prefix="Cartier")
    else:
        df = df.drop(columns=["Cartier"], errors="ignore")

    return df

def pregateste_xy_regresie(df):
    """X și y pentru predicția prețului."""
    exclude = ["Preț (EUR)", "Preț/mp (EUR)"]
    cols_x = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    df_clean = df[cols_x + ["Preț (EUR)"]].dropna()
    X = df_clean[cols_x]
    y = df_clean["Preț (EUR)"]
    return X, y, cols_x

def pregateste_xy_clasificare(df):
    """X și y pentru clasificare segment de preț (Buget/Mediu/Premium)."""
    df = df.copy()
    q33 = df["Preț (EUR)"].quantile(0.33)
    q66 = df["Preț (EUR)"].quantile(0.66)
    df["Segment"] = pd.cut(
        df["Preț (EUR)"],
        bins=[-np.inf, q33, q66, np.inf],
        labels=["Buget", "Mediu", "Premium"]
    )
    exclude = ["Preț (EUR)", "Preț/mp (EUR)", "Segment"]
    cols_x = [c for c in df.select_dtypes(include="number").columns if c not in exclude]
    df_clean = df[cols_x + ["Segment"]].dropna()
    X = df_clean[cols_x]
    y = df_clean["Segment"]
    return X, y, cols_x

def antreneaza_regresie(X, y, model_tip, n_trees, test_size, random_state):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    if model_tip == "Random Forest":
        model = RandomForestRegressor(n_estimators=n_trees, random_state=random_state)
    else:
        model = LinearRegression()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return model, X_tr, X_te, y_tr, y_te, y_pred

def antreneaza_clasificare(X, y, model_tip, n_trees, test_size, random_state):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if model_tip == "Random Forest":
        model = RandomForestClassifier(n_estimators=n_trees, random_state=random_state)
    else:
        model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    return model, X_tr, X_te, y_tr, y_te, y_pred

# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
df_raw = load_data()

st.title("Machine Learning")
st.markdown("**Concepte noi:** `st.spinner` · `st.form` avansat · `st.metric` cu delta · impact preprocesare")
st.markdown("---")

st.markdown("""
### Ce facem pe această pagină

Antrenăm **două modele** pe datele imobiliare:

- **Regresie** — predicția prețului exact în EUR
- **Clasificare** — încadrarea proprietății în segmentul *Buget / Mediu / Premium*

Apoi comparăm cum se comportă **același model** antrenat cu preprocesări diferite.
Aceasta este lecția principală a acestei pagini.
""")

# ═════════════════════════════════════════════════════════════════
# CONCEPT — st.spinner și st.form avansat
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("1. `st.spinner` și `st.form` — recap rapid")

st.markdown("""
`st.spinner` afișează un indicator de încărcare cât timp rulează o operație lentă.
Se folosește întotdeauna cu `with`:

```python
with st.spinner("Antrenez modelul..."):
    model.fit(X_train, y_train)   # operație lentă
    y_pred = model.predict(X_test)

st.success("Antrenare completă!")
```

`st.form` grupează widget-urile de configurare — scriptul nu se reexecută
la fiecare modificare, ci doar la apăsarea butonului de submit.
Combinat cu `st.spinner`, obții un flux clar: *configurează → antrenează → vezi rezultatele*.
""")

# ═════════════════════════════════════════════════════════════════
# SECȚIUNEA A — Regresie
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("2. Regresie — Predicția prețului")

st.markdown("""
**Scopul:** Modelul primește caracteristicile unei proprietăți
(suprafață, camere, cartier, an, etc.) și prezice prețul în EUR.

**Metrica principală — R²** (coeficientul de determinare):
- R² = 1.0 → model perfect
- R² = 0.0 → modelul nu explică nimic, e la fel de bun ca media
- R² < 0  → modelul e mai rău decât a prezice mereu media
""")

with st.form("form_regresie"):
    st.markdown("**Configurare model de regresie**")
    col1, col2, col3 = st.columns(3)
    with col1:
        model_reg = st.selectbox(
            "Model:", ["Random Forest", "Linear Regression"], key="sel_reg"
        )
    with col2:
        n_trees_reg = st.slider(
            "Număr arbori (RF)", 50, 300, 100, 50, key="trees_reg",
            disabled=(model_reg == "Linear Regression")
        )
    with col3:
        test_size_reg = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_reg")

    submit_reg = st.form_submit_button("Antrenează modelul de regresie")

if submit_reg:
    # Preia configurația salvată de pe Pagina 2 dacă există
    if "alegeri_preprocesare" in st.session_state:
        cfg = st.session_state["alegeri_preprocesare"]
        cfg["enc_cartier"] = cfg.get("enc_cartier", "Label Encoding")
        st.info("Folosind configurația de preprocesare aleasă pe Pagina 2.")
    else:
        cfg = {
            "metoda_an": "Median", "metoda_etaj": "Completează cu 0 (parter)",
            "metoda_outlieri": "Păstrează toți outlierii",
            "enc_tip": "Label Encoding (0/1)", "enc_oras": "Label Encoding",
            "cartier_inclus": True, "enc_cartier": "Label Encoding"
        }
        st.warning("Nu ai trecut prin Pagina 2. Folosind configurație implicită.")

    with st.spinner("Preprocesez datele și antrenez modelul..."):
        df_proc = preproceseaza(df_raw, cfg)
        X, y, cols_x = pregateste_xy_regresie(df_proc)
        model, X_tr, X_te, y_tr, y_te, y_pred = antreneaza_regresie(
            X, y, model_reg, n_trees_reg, test_size_reg, 42
        )
        r2   = r2_score(y_te, y_pred)
        rmse = np.sqrt(mean_squared_error(y_te, y_pred))
        mae  = mean_absolute_error(y_te, y_pred)

    st.success("Antrenare completă!")

    # Metrici
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R²", f"{r2:.3f}")
    col2.metric("RMSE", f"{rmse:,.0f} €")
    col3.metric("MAE", f"{mae:,.0f} €")
    col4.metric("Date antrenare", f"{len(X_tr):,}")

    # Grafice
    col1, col2 = st.columns(2)

    with col1:
        fig_pred = px.scatter(
            x=y_te, y=y_pred,
            labels={"x": "Preț real (€)", "y": "Preț prezis (€)"},
            title="Real vs. Prezis",
            color_discrete_sequence=["#2d6a4f"]
        )
        max_val = max(y_te.max(), y_pred.max())
        fig_pred.add_shape(
            type="line", x0=0, y0=0, x1=max_val, y1=max_val,
            line=dict(color="#888", dash="dash")
        )
        fig_pred.update_layout(height=380)
        st.plotly_chart(fig_pred, use_container_width=True)
        st.caption("Punctele pe linia diagonală = predicții perfecte. Cu cât mai aproape de diagonală, cu atât mai bun modelul.")

    with col2:
        erori = y_pred - y_te
        fig_err = px.histogram(
            x=erori, nbins=40,
            labels={"x": "Eroare (prezis − real) în €"},
            title="Distribuția erorilor",
            color_discrete_sequence=["#52b788"]
        )
        fig_err.add_vline(x=0, line_dash="dash", line_color="#888")
        fig_err.update_layout(height=380)
        st.plotly_chart(fig_err, use_container_width=True)
        st.caption("O distribuție centrată în 0 și îngustă indică un model bine calibrat, fără bias sistematic.")

    # Feature importance (RF)
    if model_reg == "Random Forest":
        importanta = pd.DataFrame({
            "Feature": cols_x,
            "Importanță": model.feature_importances_
        }).sort_values("Importanță", ascending=True).tail(12)

        fig_imp = px.bar(
            importanta, x="Importanță", y="Feature", orientation="h",
            title="Top 12 variabile după importanță",
            color_discrete_sequence=["#2d6a4f"]
        )
        fig_imp.update_layout(height=400)
        st.plotly_chart(fig_imp, use_container_width=True)

    # Salvăm rezultatele pentru comparație
    st.session_state["rezultat_reg"] = {
        "model": model_reg, "r2": r2, "rmse": rmse, "mae": mae,
        "cfg_label": f"{cfg['metoda_an']} | {cfg['metoda_outlieri']}"
    }

# ── Exercițiu 1 ───────────────────────────────────────────────────
st.subheader("Exercițiul 1")
st.markdown("""
Antrenează modelul de regresie de mai sus, notează valorile R², RMSE și MAE.

Apoi mergi pe **Pagina 2**, schimbă metoda de imputare pentru `An construcție`
din **Median** în **KNN Imputer**, întoarce-te și antrenează din nou.

Răspunde în căsuța de mai jos:
- S-a schimbat R²? Cu cât?
- Ce metodă de imputare a dat rezultate mai bune?
- De ce crezi că există o diferență?
""")

with st.expander("Indiciu"):
    st.markdown("""
    R² măsoară proporția din varianța prețului explicată de model.
    KNN Imputer estimează valorile lipsă pe baza proprietăților similare din dataset,
    ceea ce poate produce valori mai precise decât o simplă mediană globală.
    Gândește-te: dacă un apartament din Floreasca are `An construcție` lipsă,
    e mai logic să îl estimezi cu mediana tuturor proprietăților sau
    cu media celor mai similare proprietăți din vecinătate?
    """)

raspuns_ex1 = st.text_area(
    "Observațiile tale:",
    value="",
    height=100,
    placeholder="Scrie aici ce ai observat după cele două antrenări...",
    key="obs_ex1"
)

# ═════════════════════════════════════════════════════════════════
# SECȚIUNEA B — Clasificare
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("3. Clasificare — Segment de preț")

st.markdown("""
**Scopul:** Modelul clasifică fiecare proprietate într-unul din trei segmente
calculate automat pe baza distribuției prețurilor din dataset:

| Segment | Interval |
|---|---|
| **Buget** | Sub percentila 33% |
| **Mediu** | Între percentilele 33% și 66% |
| **Premium** | Peste percentila 66% |

**Metrica principală — Accuracy:** proporția predicțiilor corecte din total.
""")

with st.form("form_clasificare"):
    st.markdown("**Configurare model de clasificare**")
    col1, col2, col3 = st.columns(3)
    with col1:
        model_cls = st.selectbox(
            "Model:", ["Random Forest", "Logistic Regression"], key="sel_cls"
        )
    with col2:
        n_trees_cls = st.slider(
            "Număr arbori (RF)", 50, 300, 100, 50, key="trees_cls",
            disabled=(model_cls == "Logistic Regression")
        )
    with col3:
        test_size_cls = st.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="ts_cls")

    submit_cls = st.form_submit_button("Antrenează modelul de clasificare")

if submit_cls:
    if "alegeri_preprocesare" in st.session_state:
        cfg = st.session_state["alegeri_preprocesare"]
        cfg["enc_cartier"] = cfg.get("enc_cartier", "Label Encoding")
        st.info("Folosind configurația de preprocesare aleasă pe Pagina 2.")
    else:
        cfg = {
            "metoda_an": "Median", "metoda_etaj": "Completează cu 0 (parter)",
            "metoda_outlieri": "Păstrează toți outlierii",
            "enc_tip": "Label Encoding (0/1)", "enc_oras": "Label Encoding",
            "cartier_inclus": True, "enc_cartier": "Label Encoding"
        }
        st.warning("Nu ai trecut prin Pagina 2. Folosind configurație implicită.")

    with st.spinner("Preprocesez datele și antrenez modelul..."):
        df_proc = preproceseaza(df_raw, cfg)
        X, y, cols_x = pregateste_xy_clasificare(df_proc)
        model, X_tr, X_te, y_tr, y_te, y_pred = antreneaza_clasificare(
            X, y, model_cls, n_trees_cls, test_size_cls, 42
        )
        acc = accuracy_score(y_te, y_pred)

    st.success("Antrenare completă!")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.3f}")
    col2.metric("Date antrenare", f"{len(X_tr):,}")
    col3.metric("Date testare", f"{len(X_te):,}")

    col1, col2 = st.columns(2)

    with col1:
        # Confusion matrix
        etichete = ["Buget", "Mediu", "Premium"]
        cm = confusion_matrix(y_te, y_pred, labels=etichete)
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=etichete, y=etichete,
            labels=dict(x="Prezis", y="Real"),
            title="Matrice de confuzie",
            color_continuous_scale=[[0, "#f0ebe3"], [1, "#2d6a4f"]]
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)
        st.caption("Diagonala principală = predicții corecte. Valorile în afara diagonalei = erori de clasificare.")

    with col2:
        # Distribuție segmente
        dist = pd.DataFrame({
            "Segment": etichete,
            "Real": [sum(y_te == s) for s in etichete],
            "Prezis": [sum(y_pred == s) for s in etichete]
        }).melt(id_vars="Segment", var_name="Tip", value_name="Count")

        fig_dist = px.bar(
            dist, x="Segment", y="Count", color="Tip", barmode="group",
            title="Distribuție segmente — real vs. prezis",
            color_discrete_sequence=["#2d6a4f", "#52b788"]
        )
        fig_dist.update_layout(height=380)
        st.plotly_chart(fig_dist, use_container_width=True)

    # Raport clasificare
    with st.expander("Raport detaliat per segment"):
        raport = classification_report(y_te, y_pred, labels=etichete, output_dict=True)
        df_raport = pd.DataFrame(raport).transpose().round(3)
        st.dataframe(df_raport, use_container_width=True)
        st.markdown("""
        - **precision** — din toate predicțiile pentru un segment, câte sunt corecte
        - **recall** — din toate cazurile reale dintr-un segment, câte le-a găsit modelul
        - **f1-score** — media armonică dintre precision și recall
        """)

    st.session_state["rezultat_cls"] = {
        "model": model_cls, "acc": acc,
        "cfg_label": f"{cfg['metoda_an']} | {cfg['metoda_outlieri']}"
    }

# ── Exercițiu 2 ───────────────────────────────────────────────────
st.subheader("Exercițiul 2")
st.markdown("""
Antrenează modelul de clasificare cu **Random Forest**.
Apoi mergi pe Pagina 2 și schimbă tratarea outlierilor din
**Păstrează toți outlierii** în **Elimină rândurile outlieri**.

Întoarce-te și antrenează din nou. Observi o diferență în Accuracy?
Uită-te și la matricea de confuzie — s-a schimbat ceva pentru segmentul **Premium**?
""")

with st.expander("Indiciu"):
    st.markdown("""
    Outlierii din dataset sunt în mare parte proprietăți de lux cu prețuri foarte mari —
    exact segmentul Premium. Când îi elimini, modelul are mai puține exemple
    din care să învețe cum arată o proprietate Premium.
    Gândește-te cum afectează asta recall-ul pentru clasa Premium din raportul detaliat.
    """)

raspuns_ex2 = st.text_area(
    "Observațiile tale:",
    value="",
    height=100,
    placeholder="Ce ai observat? S-a schimbat accuracy? Dar clasificarea pentru Premium?",
    key="obs_ex2"
)

# ═════════════════════════════════════════════════════════════════
# SECȚIUNEA C — Comparație directă preprocesări
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("4. Comparație directă — același model, preprocesări diferite")

st.markdown("""
Aceasta este secțiunea cheie a seminarului. Antrenăm același model
de **trei ori**, cu trei configurații de preprocesare diferite,
și comparăm rezultatele side-by-side.
""")

with st.form("form_comparatie"):
    st.markdown("**Model fix pentru comparație**")
    col1, col2 = st.columns(2)
    with col1:
        model_comp = st.selectbox(
            "Model:", ["Random Forest", "Linear Regression"], key="sel_comp"
        )
        tip_task = st.radio(
            "Task:", ["Regresie (preț)", "Clasificare (segment)"],
            horizontal=True, key="task_comp"
        )
    with col2:
        n_trees_comp = st.slider("Număr arbori", 50, 200, 100, 50, key="trees_comp")

    st.markdown("---")
    st.markdown("**Configurație A**")
    col1, col2, col3 = st.columns(3)
    with col1:
        a_imp = st.selectbox("Imputare An:", ["Mean", "Median", "KNN Imputer"], key="a_imp")
    with col2:
        a_out = st.selectbox("Outlieri:", [
            "Păstrează toți outlierii", "Elimină rândurile outlieri",
            "Capping la percentile"
        ], key="a_out")
    with col3:
        a_enc = st.selectbox("Encoding Oraș:", ["Label Encoding", "One-Hot Encoding"], key="a_enc")

    st.markdown("**Configurație B**")
    col1, col2, col3 = st.columns(3)
    with col1:
        b_imp = st.selectbox("Imputare An:", ["Median", "Mean", "KNN Imputer",
                                              "Forward Fill (ffill)"], key="b_imp")
    with col2:
        b_out = st.selectbox("Outlieri:", [
            "Elimină rândurile outlieri", "Păstrează toți outlierii",
            "Capping la percentile"
        ], key="b_out")
    with col3:
        b_enc = st.selectbox("Encoding Oraș:", ["One-Hot Encoding", "Label Encoding"], key="b_enc")

    st.markdown("**Configurație C**")
    col1, col2, col3 = st.columns(3)
    with col1:
        c_imp = st.selectbox("Imputare An:", ["Forward Fill (ffill)", "Mean",
                                              "Median", "KNN Imputer"], key="c_imp")
    with col2:
        c_out = st.selectbox("Outlieri:", [
            "Capping la percentile", "Păstrează toți outlierii",
            "Elimină rândurile outlieri"
        ], key="c_out")
    with col3:
        c_enc = st.selectbox("Encoding Oraș:", ["Label Encoding", "One-Hot Encoding"], key="c_enc")

    submit_comp = st.form_submit_button("Compară cele 3 configurații")

if submit_comp:
    cfg_base = {
        "metoda_etaj": "Completează cu 0 (parter)",
        "enc_tip": "Label Encoding (0/1)",
        "cartier_inclus": True,
        "enc_cartier": "Label Encoding"
    }

    configs = [
        {"label": "A", "metoda_an": a_imp, "metoda_outlieri": a_out, "enc_oras": a_enc},
        {"label": "B", "metoda_an": b_imp, "metoda_outlieri": b_out, "enc_oras": b_enc},
        {"label": "C", "metoda_an": c_imp, "metoda_outlieri": c_out, "enc_oras": c_enc},
    ]

    rezultate = []
    progress = st.progress(0, text="Se antrenează configurația 1/3...")

    for i, cfg in enumerate(configs):
        cfg_full = {**cfg_base, **cfg}
        df_proc = preproceseaza(df_raw, cfg_full)

        if tip_task == "Regresie (preț)":
            X, y, _ = pregateste_xy_regresie(df_proc)
            model, X_tr, X_te, y_tr, y_te, y_pred = antreneaza_regresie(
                X, y, model_comp, n_trees_comp, 0.2, 42
            )
            r2   = r2_score(y_te, y_pred)
            rmse = np.sqrt(mean_squared_error(y_te, y_pred))
            rezultate.append({
                "Configurație": cfg["label"],
                "Imputare An": cfg["metoda_an"],
                "Outlieri": cfg["metoda_outlieri"],
                "Encoding Oraș": cfg["enc_oras"],
                "Rânduri": len(X),
                "R²": round(r2, 4),
                "RMSE (€)": int(rmse),
            })
        else:
            X, y, _ = pregateste_xy_clasificare(df_proc)
            model, X_tr, X_te, y_tr, y_te, y_pred = antreneaza_clasificare(
                X, y, model_comp, n_trees_comp, 0.2, 42
            )
            acc = accuracy_score(y_te, y_pred)
            rezultate.append({
                "Configurație": cfg["label"],
                "Imputare An": cfg["metoda_an"],
                "Outlieri": cfg["metoda_outlieri"],
                "Encoding Oraș": cfg["enc_oras"],
                "Rânduri": len(X),
                "Accuracy": round(acc, 4),
            })

        progress.progress((i + 1) / 3, text=f"Configurația {i+1}/3 completă.")

    df_rez = pd.DataFrame(rezultate)
    st.success("Comparație completă!")
    st.dataframe(df_rez, use_container_width=True, hide_index=True)

    # Grafic comparativ
    if tip_task == "Regresie (preț)":
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(
                df_rez, x="Configurație", y="R²",
                title="R² — mai mare e mai bun",
                color="Configurație",
                color_discrete_sequence=["#2d6a4f", "#52b788", "#b7e4c7"],
                text="R²"
            )
            fig.update_traces(textposition="outside")
            fig.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.bar(
                df_rez, x="Configurație", y="RMSE (€)",
                title="RMSE — mai mic e mai bun",
                color="Configurație",
                color_discrete_sequence=["#2d6a4f", "#52b788", "#b7e4c7"],
                text="RMSE (€)"
            )
            fig2.update_traces(textposition="outside")
            fig2.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        fig = px.bar(
            df_rez, x="Configurație", y="Accuracy",
            title="Accuracy — mai mare e mai bun",
            color="Configurație",
            color_discrete_sequence=["#2d6a4f", "#52b788", "#b7e4c7"],
            text="Accuracy"
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Cea mai bună configurație
    if tip_task == "Regresie (preț)":
        best = df_rez.loc[df_rez["R²"].idxmax()]
        st.info(f"**Cea mai bună configurație: {best['Configurație']}** — R² = {best['R²']}, RMSE = {best['RMSE (€)']:,} €")
    else:
        best = df_rez.loc[df_rez["Accuracy"].idxmax()]
        st.info(f"**Cea mai bună configurație: {best['Configurație']}** — Accuracy = {best['Accuracy']}")

# ── Exercițiu 3 ───────────────────────────────────────────────────
st.subheader("Exercițiul 3")
st.markdown("""
Folosind comparatorul de mai sus, găsește combinația care produce cel mai bun R²
pentru modelul **Random Forest** pe task-ul de **regresie**.

Poți varia liber imputarea, tratarea outlierilor și encoding-ul.
Notează combinația câștigătoare și explică de ce crezi că funcționează mai bine.
""")

with st.expander("Indiciu"):
    st.markdown("""
    Nu există un răspuns universal corect — depinde de dataset.
    Încearcă să gândești logic: KNN Imputer folosește informații din date
    pentru a estima valorile lipsă, ceea ce poate fi mai precis decât Mean/Median.
    Eliminarea outlierilor reduce zgomotul, dar și informația despre proprietățile scumpe.
    One-Hot Encoding pentru `Oraș` creează coloane separate per oraș —
    modelul poate învăța diferențe mai nuanțate față de Label Encoding.
    """)

raspuns_ex3 = st.text_area(
    "Combinația câștigătoare și explicația ta:",
    value="",
    height=120,
    placeholder="Imputare: ... | Outlieri: ... | Encoding: ... \nMotivul: ...",
    key="obs_ex3"
)

# ── Rezumat ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Rezumat")
st.markdown("""
**Ce am demonstrat pe această pagină:**
- Același model antrenat cu preprocesări diferite produce rezultate diferite
- Imputarea valorilor lipsă influențează calitatea datelor de intrare în model
- Eliminarea outlierilor poate îmbunătăți sau înrăutăți performanța — depinde de context
- Encoding-ul variabilelor categorice afectează cum vede modelul relațiile dintre date

**Concluzie:** Preprocesarea nu este un pas tehnic de bifat.
Este o serie de **decizii** care trebuie justificate și validate experimental.
""")

st.info("Continuă cu **Pagina 4 — Quiz** pentru a-ți testa cunoștințele.")