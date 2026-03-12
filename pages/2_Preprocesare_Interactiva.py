import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder

@st.cache_data
def load_data():
    return pd.read_csv("imobiliare_romania.csv")

df_original = load_data()

st.title("Preprocesare Interactivă")
st.markdown("**Widget-uri noi:** `st.progress` · `st.data_editor` · `st.selectbox` avansat")
st.markdown("---")

st.markdown("""
### De ce contează alegerile din preprocesare?

Datele brute conțin aproape întotdeauna **valori lipsă**, **outlieri** și
**variabile categoriale**. Cum le tratezi influențează direct calitatea modelului ML.

Pe această pagină vei lua decizii concrete de preprocesare.
Pe **Pagina 3** vei vedea cum aceste decizii afectează performanța modelului.
""")

st.info("""
**Atenție:** Fiecare secțiune aplică transformările în ordine.
Datele prelucrate la pasul 1 sunt folosite la pasul 2, și așa mai departe.
Urmărește cum se schimbă statisticile pe parcurs.
""")

# ═════════════════════════════════════════════════════════════════
# PASUL 1 — Valori lipsă
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Pasul 1 — Tratarea valorilor lipsă")

st.markdown("""
`st.progress` afișează o bară de progres — util pentru operații în mai mulți pași
sau pentru a vizualiza proporții. Acceptă o valoare între `0.0` și `1.0`.

```python
st.progress(0.75, text="75% completat")
```
""")

st.subheader("Situația actuală")

col1, col2 = st.columns(2)
with col1:
    lipsa = df_original.isnull().sum()
    lipsa = lipsa[lipsa > 0].reset_index()
    lipsa.columns = ["Coloană", "Valori lipsă"]
    lipsa["Procent (%)"] = (lipsa["Valori lipsă"] / len(df_original) * 100).round(1)
    st.dataframe(lipsa, use_container_width=True, hide_index=True)

with col2:
    for _, row in lipsa.iterrows():
        st.markdown(f"**{row['Coloană']}**")
        st.progress(
            row["Procent (%)"] / 100,
            text=f"{row['Valori lipsă']} valori lipsă ({row['Procent (%)']}%)"
        )
        st.markdown("")

st.markdown("""
**Contextul valorilor lipsă din dataset:**
- `Etaj` — casele nu au etaj. Valoarea lipsă nu este o eroare, ci o caracteristică.
  Vom trata această coloană separat față de `An construcție`.
- `An construcție` — informație nedisponibilă în anunț. Aceasta este o lipsă reală.
""")

st.subheader("Alege metoda de imputare")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**`An construcție`** — lipsă reală, coloană numerică")
    metoda_an = st.selectbox(
        "Metoda pentru An construcție:",
        ["Mean", "Median", "Forward Fill (ffill)", "Backward Fill (bfill)", "KNN Imputer"],
        key="metoda_an"
    )

with col2:
    st.markdown("**`Etaj`** — lipsă structurală (case), coloană numerică")
    metoda_etaj = st.selectbox(
        "Metoda pentru Etaj:",
        ["Completează cu 0 (parter)", "Median", "KNN Imputer", "Lasă lipsă"],
        key="metoda_etaj"
    )

with st.expander("Cum funcționează fiecare metodă?"):
    st.markdown("""
    - **Mean / Median** — înlocuiește valorile lipsă cu media sau mediana coloanei.
      Media e sensibilă la outlieri; mediana e mai robustă.
    - **Forward Fill (ffill)** — copiază ultima valoare validă de deasupra.
      Are sens când datele au o ordine logică (timp, proximitate geografică).
    - **Backward Fill (bfill)** — copiază prima valoare validă de dedesubt.
      Similar cu ffill, dar în direcție inversă.
    - **KNN Imputer** — estimează valoarea lipsă pe baza celor mai similari
      `k` vecini din dataset. Mai precis, dar mai lent.
    - **Completează cu 0** — adecvat când 0 are o semnificație clară
      (în cazul nostru: parter / casă fără etaj).
    """)

# Aplicare imputare
df_pas1 = df_original.copy()

# An construcție
if metoda_an == "Mean":
    imp = SimpleImputer(strategy="mean")
    df_pas1["An construcție"] = imp.fit_transform(df_pas1[["An construcție"]]).ravel().round().astype(float)
elif metoda_an == "Median":
    imp = SimpleImputer(strategy="median")
    df_pas1["An construcție"] = imp.fit_transform(df_pas1[["An construcție"]]).ravel().round().astype(float)
elif metoda_an == "Forward Fill (ffill)":
    df_pas1["An construcție"] = df_pas1["An construcție"].ffill().round().astype(float)
elif metoda_an == "Backward Fill (bfill)":
    df_pas1["An construcție"] = df_pas1["An construcție"].bfill().round().astype(float)
elif metoda_an == "KNN Imputer":
    knn = KNNImputer(n_neighbors=5)
    cols_knn = ["An construcție", "Suprafață (mp)", "Preț (EUR)"]
    df_pas1[cols_knn] = knn.fit_transform(df_pas1[cols_knn])
    df_pas1["An construcție"] = df_pas1["An construcție"].round().astype(float)

# Etaj
if metoda_etaj == "Completează cu 0 (parter)":
    df_pas1["Etaj"] = df_pas1["Etaj"].fillna(0).astype(float)
elif metoda_etaj == "Median":
    med = df_pas1["Etaj"].median()
    df_pas1["Etaj"] = df_pas1["Etaj"].fillna(med).astype(float)
elif metoda_etaj == "KNN Imputer":
    knn2 = KNNImputer(n_neighbors=5)
    cols_knn2 = ["Etaj", "Suprafață (mp)", "Nr. camere"]
    df_pas1[cols_knn2] = knn2.fit_transform(df_pas1[cols_knn2])
    df_pas1["Etaj"] = df_pas1["Etaj"].round().astype(float)
# "Lasă lipsă" — nu facem nimic

lipsa_dupa = df_pas1.isnull().sum().sum()
if metoda_etaj == "Lasă lipsă":
    st.warning(f"Au rămas **{lipsa_dupa}** valori lipsă după imputare (Etaj lăsat neatins).")
else:
    st.success(f"Valori lipsă după imputare: **{lipsa_dupa}**")

col1, col2, col3 = st.columns(3)
col1.metric("An construcție — medie", f"{df_pas1['An construcție'].mean():.0f}")
col2.metric("An construcție — înainte", f"{df_original['An construcție'].mean():.0f}")
col3.metric(
    "Diferență medie",
    f"{df_pas1['An construcție'].mean() - df_original['An construcție'].mean():.1f} ani",
    delta_color="off"
)

# ── Exercițiu 1 ───────────────────────────────────────────────────
st.subheader("Exercițiul 1")
st.markdown("""
Folosind `st.progress`, vizualizează **completitudinea fiecărei coloane** din `df_pas1`
— adică procentul de valori NON-lipsă (completitudinea, nu lipsa).

O coloană fără nicio valoare lipsă ar trebui să afișeze o bară plină (1.0).
""")

with st.expander("Indiciu"):
    st.markdown("""
    Completitudinea unei coloane se calculează ca numărul de valori non-nule
    împărțit la numărul total de rânduri. Funcția `.notnull().sum()` numără
    valorile non-nule, iar `.shape[0]` îți dă numărul total de rânduri.
    Iterează prin coloane cu un `for` și apelează `st.progress()` pentru fiecare.
    """)

cod_ex1 = st.text_area(
    "Codul tău:",
    value="# Afișează completitudinea fiecărei coloane cu st.progress\n",
    height=130,
    key="ex1_code"
)
if st.button("Run", key="ex1_run"):
    try:
        exec(cod_ex1, {"st": st, "pd": pd, "np": np, "df_pas1": df_pas1})
    except Exception as e:
        st.error(f"Eroare: {e}")

# ═════════════════════════════════════════════════════════════════
# PASUL 2 — Outlieri
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Pasul 2 — Tratarea outlierilor")

st.markdown("""
Outlierii sunt valori extreme care pot distorsiona modelul ML.
Există două abordări principale: **eliminarea** sau **înlocuirea** (capping).

```python
# Metoda IQR — cel mai frecvent folosită
Q1  = df["coloana"].quantile(0.25)
Q3  = df["coloana"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
```
""")

st.subheader("Explorează distribuția prețurilor")

col1, col2 = st.columns([2, 1])
with col1:
    fig_box = px.box(
        df_pas1, x="Oraș", y="Preț (EUR)", color="Tip",
        title="Boxplot Preț — identificare outlieri",
        color_discrete_sequence=["#2d6a4f", "#52b788"]
    )
    fig_box.update_layout(height=350)
    st.plotly_chart(fig_box, use_container_width=True)

with col2:
    Q1 = df_pas1["Preț (EUR)"].quantile(0.25)
    Q3 = df_pas1["Preț (EUR)"].quantile(0.75)
    IQR = Q3 - Q1
    n_outlieri = len(df_pas1[
        (df_pas1["Preț (EUR)"] < Q1 - 1.5 * IQR) |
        (df_pas1["Preț (EUR)"] > Q3 + 1.5 * IQR)
    ])
    st.metric("Q1 (25%)", f"{Q1:,.0f} €")
    st.metric("Q3 (75%)", f"{Q3:,.0f} €")
    st.metric("IQR", f"{IQR:,.0f} €")
    st.metric("Outlieri detectați (IQR)", f"{n_outlieri}")

st.subheader("Alege metoda de tratare a outlierilor")

col1, col2 = st.columns(2)
with col1:
    metoda_out = st.radio(
        "Metodă:",
        ["Elimină rândurile outlieri", "Capping la percentile", "Păstrează toți outlierii"],
        key="metoda_out"
    )

with col2:
    if metoda_out == "Capping la percentile":
        percentil_jos = st.slider("Percentilă inferioară", 0, 10, 1, key="perc_jos")
        percentil_sus = st.slider("Percentilă superioară", 90, 100, 99, key="perc_sus")
    else:
        st.markdown("")

df_pas2 = df_pas1.copy()

if metoda_out == "Elimină rândurile outlieri":
    Q1_ = df_pas2["Preț (EUR)"].quantile(0.25)
    Q3_ = df_pas2["Preț (EUR)"].quantile(0.75)
    IQR_ = Q3_ - Q1_
    df_pas2 = df_pas2[
        (df_pas2["Preț (EUR)"] >= Q1_ - 1.5 * IQR_) &
        (df_pas2["Preț (EUR)"] <= Q3_ + 1.5 * IQR_)
    ].reset_index(drop=True)

elif metoda_out == "Capping la percentile":
    low  = df_pas2["Preț (EUR)"].quantile(percentil_jos / 100)
    high = df_pas2["Preț (EUR)"].quantile(percentil_sus / 100)
    df_pas2["Preț (EUR)"] = df_pas2["Preț (EUR)"].clip(low, high)
    df_pas2["Preț/mp (EUR)"] = (df_pas2["Preț (EUR)"] / df_pas2["Suprafață (mp)"]).round()

col1, col2, col3 = st.columns(3)
col1.metric("Rânduri înainte", len(df_pas1))
col2.metric("Rânduri după", len(df_pas2))
col3.metric("Rânduri eliminate", len(df_pas1) - len(df_pas2))

# ── Exercițiu 2 ───────────────────────────────────────────────────
st.subheader("Exercițiul 2")
st.markdown("""
Construiește un **histogram comparativ** — înainte și după tratarea outlierilor —
pentru coloana `Preț/mp (EUR)`.

Folosește `st.columns(2)` pentru a le afișa side-by-side și
adaugă câte un `st.metric` sub fiecare grafic cu prețul mediu/mp.
""")

with st.expander("Indiciu"):
    st.markdown("""
    Ai la dispoziție `df_pas1` (înainte) și `df_pas2` (după).
    Creează două coloane cu `st.columns(2)`, pune câte un `px.histogram`
    în fiecare și câte un `st.metric` cu media coloanei `Preț/mp (EUR)`.
    """)

cod_ex2 = st.text_area(
    "Codul tău:",
    value="# Histogram comparativ înainte/după pentru Preț/mp\n",
    height=140,
    key="ex2_code"
)
if st.button("Run", key="ex2_run"):
    try:
        exec(cod_ex2, {
            "st": st, "pd": pd, "np": np, "px": px,
            "df_pas1": df_pas1, "df_pas2": df_pas2
        })
    except Exception as e:
        st.error(f"Eroare: {e}")

# ═════════════════════════════════════════════════════════════════
# PASUL 3 — Encoding variabile categoriale
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Pasul 3 — Encoding variabile categoriale")

st.markdown("""
Modelele ML nu înțeleg text — variabilele categoriale trebuie convertite în numere.
Există două metode principale:

| Metodă | Când se folosește | Dezavantaj |
|---|---|---|
| **Label Encoding** | Variabile ordinale sau binare | Introduce ordine artificială |
| **One-Hot Encoding** | Variabile nominale fără ordine | Crește numărul de coloane |

Pentru `Tip` (Apartament/Casă) — variabilă binară → Label Encoding are sens.
Pentru `Oraș` și `Cartier` — variabile nominale → One-Hot Encoding e mai corect.
""")

st.subheader("Configurează encoding-ul")

col1, col2, col3 = st.columns(3)
with col1:
    enc_tip = st.selectbox(
        "Encoding pentru `Tip`:",
        ["Label Encoding (0/1)", "One-Hot Encoding"],
        key="enc_tip"
    )
with col2:
    enc_oras = st.selectbox(
        "Encoding pentru `Oraș`:",
        ["Label Encoding", "One-Hot Encoding"],
        key="enc_oras"
    )
with col3:
    inc_cartier = st.checkbox(
        "Include `Cartier` în model?",
        value=True,
        key="inc_cartier"
    )
    if inc_cartier:
        enc_cartier = st.selectbox(
            "Encoding pentru `Cartier`:",
            ["Label Encoding", "One-Hot Encoding"],
            key="enc_cartier"
        )

df_pas3 = df_pas2.copy()

# Tip
if enc_tip == "Label Encoding (0/1)":
    df_pas3["Tip"] = (df_pas3["Tip"] == "Apartament").astype(int)
else:
    df_pas3 = pd.get_dummies(df_pas3, columns=["Tip"], prefix="Tip")

# Oraș
if enc_oras == "Label Encoding":
    le = LabelEncoder()
    df_pas3["Oraș"] = le.fit_transform(df_pas3["Oraș"])
else:
    df_pas3 = pd.get_dummies(df_pas3, columns=["Oraș"], prefix="Oraș")

# Cartier
if inc_cartier:
    if enc_cartier == "Label Encoding":
        le2 = LabelEncoder()
        df_pas3["Cartier"] = le2.fit_transform(df_pas3["Cartier"])
    else:
        df_pas3 = pd.get_dummies(df_pas3, columns=["Cartier"], prefix="Cartier")
else:
    df_pas3 = df_pas3.drop(columns=["Cartier"])

col1, col2 = st.columns(2)
col1.metric("Coloane înainte de encoding", len(df_pas2.columns))
col2.metric("Coloane după encoding", len(df_pas3.columns))

st.markdown("**Primele rânduri după encoding:**")
st.dataframe(df_pas3.head(5), use_container_width=True, hide_index=True)

# ── Exercițiu 3 ───────────────────────────────────────────────────
st.subheader("Exercițiul 3")
st.markdown("""
După encoding, unele coloane au valori neașteptate sau tipuri greșite.
Folosește `st.data_editor` pentru a afișa primele **10 rânduri** din `df_pas3`
și permite editarea manuală.

`st.data_editor` funcționează ca `st.dataframe`, dar permite modificarea
valorilor direct în tabel:

```python
df_editat = st.data_editor(df, num_rows="dynamic")
```

Afișează și numărul de coloane numerice din `df_pas3` folosind `st.metric`.
""")

with st.expander("Indiciu"):
    st.markdown("""
    `st.data_editor` returnează un DataFrame modificat — poți salva rezultatul
    într-o variabilă nouă. Pentru a număra coloanele numerice, folosește
    `.select_dtypes(include="number").shape[1]`.
    """)

cod_ex3 = st.text_area(
    "Codul tău:",
    value="# Folosește st.data_editor pentru df_pas3 și afișează un metric\n",
    height=130,
    key="ex3_code"
)
if st.button("Run", key="ex3_run"):
    try:
        exec(cod_ex3, {"st": st, "pd": pd, "np": np, "df_pas3": df_pas3})
    except Exception as e:
        st.error(f"Eroare: {e}")

# ═════════════════════════════════════════════════════════════════
# SUMAR — alegerile tale
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Sumar — alegerile tale de preprocesare")

st.markdown("Acestea sunt deciziile pe care le-ai luat. Pe Pagina 3 vei vedea cum influențează modelul.")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Pasul 1 — Valori lipsă**")
    st.markdown(f"- `An construcție`: **{metoda_an}**")
    st.markdown(f"- `Etaj`: **{metoda_etaj}**")

with col2:
    st.markdown("**Pasul 2 — Outlieri**")
    st.markdown(f"- Metodă: **{metoda_out}**")
    st.markdown(f"- Rânduri rămase: **{len(df_pas2)}** din {len(df_original)}")

with col3:
    st.markdown("**Pasul 3 — Encoding**")
    st.markdown(f"- `Tip`: **{enc_tip}**")
    st.markdown(f"- `Oraș`: **{enc_oras}**")
    st.markdown(f"- `Cartier`: **{'inclus — ' + enc_cartier if inc_cartier else 'exclus'}**")
    st.markdown(f"- Coloane finale: **{len(df_pas3.columns)}**")

st.markdown("<br>", unsafe_allow_html=True)

# Salvăm în session_state pentru Pagina 3
st.session_state["df_procesat"] = df_pas3
st.session_state["alegeri_preprocesare"] = {
    "metoda_an": metoda_an,
    "metoda_etaj": metoda_etaj,
    "metoda_outlieri": metoda_out,
    "enc_tip": enc_tip,
    "enc_oras": enc_oras,
    "cartier_inclus": inc_cartier,
    "n_randuri": len(df_pas3),
    "n_coloane": len(df_pas3.columns)
}

st.success("Datele preprocesate au fost salvate. Navighează la **Pagina 3 — Machine Learning** pentru a antrena modelele.")