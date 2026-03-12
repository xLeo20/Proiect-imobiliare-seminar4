import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ── Dataset ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("imobiliare_romania.csv")

df = load_data()

# ── Header ────────────────────────────────────────────────────────
st.title("Interactivitate Avansată")
st.markdown("**Widget-uri noi:** `st.radio` · `st.checkbox` · `st.form` · `st.download_button`")
st.markdown("---")

st.info("""
**Cum funcționează această pagină — pattern-ul Learn → Code → Verify**

Fiecare secțiune are trei părți:
1. **Concept** — explicația widget-ului cu sintaxă și exemple
2. **Demo** — widget-ul în acțiune pe datele imobiliare
3. **Exercițiu** — scrii tu codul în căsuța de mai jos și apeși Run
""")

# ═════════════════════════════════════════════════════════════════
# 1. st.radio
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("1. Butoane radio — `st.radio`")

st.markdown("""
`st.radio` creează un grup de butoane din care utilizatorul poate alege **o singură opțiune**.
Spre deosebire de `st.selectbox` (dropdown), toate opțiunile sunt vizibile simultan.

```python
alegere = st.radio(
    label="Alege o opțiune:",
    options=["Opțiunea A", "Opțiunea B", "Opțiunea C"],
    index=0,           # opțiunea selectată implicit (0 = prima)
    horizontal=True    # afișare pe orizontală în loc de verticală
)
st.write(f"Ai ales: {alegere}")
```

**Când folosești `st.radio` vs `st.selectbox`?**
- `st.radio` — când ai **2–4 opțiuni** și vrei să fie toate vizibile
- `st.selectbox` — când ai **mai multe opțiuni** sau spațiu limitat
""")

st.subheader("Demo: Vizualizare date după oraș")

oras_ales = st.radio(
    "Selectează orașul:",
    options=["Ambele", "București", "Cluj-Napoca"],
    horizontal=True,
    key="radio_demo"
)

if oras_ales == "Ambele":
    df_viz = df
else:
    df_viz = df[df["Oraș"] == oras_ales]

col1, col2 = st.columns(2)
with col1:
    st.metric("Anunțuri", f"{len(df_viz):,}")
    st.metric("Preț mediu", f"{df_viz['Preț (EUR)'].mean():,.0f} €")
with col2:
    st.metric("Preț/mp mediu", f"{df_viz['Preț/mp (EUR)'].mean():,.0f} €/mp")
    st.metric("Suprafață medie", f"{df_viz['Suprafață (mp)'].mean():.0f} mp")

fig = px.histogram(
    df_viz, x="Preț (EUR)", nbins=40,
    color="Oraș" if oras_ales == "Ambele" else None,
    title=f"Distribuția prețurilor — {oras_ales}",
    color_discrete_sequence=["#2d6a4f", "#52b788"]
)
fig.update_layout(height=320)
st.plotly_chart(fig, use_container_width=True)

# ── Exercițiu 1 ───────────────────────────────────────────────────
st.subheader("Exercițiul 1")
st.markdown("""
Creează un `st.radio` cu opțiunile `"Apartament"` și `"Casă"`.
În funcție de alegere, afișează câte proprietăți de acel tip există în dataset
și care este prețul mediu, folosind `st.metric`.
""")

with st.expander("Indiciu"):
    st.markdown("""
    Filtrează `df` pe coloana `"Tip"` folosind valoarea returnată de `st.radio`.
    Numărul de proprietăți este lungimea DataFrame-ului filtrat.
    Prețul mediu se calculează cu `.mean()` pe coloana `"Preț (EUR)"`.
    """)

cod_ex1 = st.text_area(
    "Codul tău:",
    value="# Creează un st.radio pentru tipul de proprietate\n# Afișează numărul și prețul mediu cu st.metric\n",
    height=140,
    key="ex1_code"
)
if st.button("Run", key="ex1_run"):
    try:
        exec(cod_ex1, {"st": st, "pd": pd, "np": np, "df": df, "px": px})
    except Exception as e:
        st.error(f"Eroare: {e}")

# ═════════════════════════════════════════════════════════════════
# 2. st.checkbox
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("2. Casete de bifare — `st.checkbox`")

st.markdown("""
`st.checkbox` returnează `True` sau `False` și este ideal pentru a **activa/dezactiva**
afișarea unui element sau a aplica un filtru opțional.

```python
arata_date = st.checkbox("Afișează datele brute")

if arata_date:
    st.dataframe(df)
```

Poți folosi mai multe checkbox-uri independente pentru filtre cumulative:

```python
filtru_buc  = st.checkbox("București")
filtru_cluj = st.checkbox("Cluj-Napoca")

if filtru_buc and filtru_cluj:
    st.write("Ambele orașe selectate")
elif filtru_buc:
    st.write("Doar București")
```
""")

st.subheader("Demo: Filtre cumulative cu checkbox")

col1, col2, col3 = st.columns(3)
with col1:
    cb_buc  = st.checkbox("București",   value=True, key="cb_buc")
    cb_cluj = st.checkbox("Cluj-Napoca", value=True, key="cb_cluj")
with col2:
    cb_apart = st.checkbox("Apartamente", value=True, key="cb_apart")
    cb_casa  = st.checkbox("Case",        value=True, key="cb_casa")
with col3:
    arata_outlieri = st.checkbox("Afișează outlieri", value=False, key="cb_out")
    arata_tabel    = st.checkbox("Afișează tabelul filtrat", value=False, key="cb_tabel")

orase_sel = []
if cb_buc:  orase_sel.append("București")
if cb_cluj: orase_sel.append("Cluj-Napoca")

tipuri_sel = []
if cb_apart: tipuri_sel.append("Apartament")
if cb_casa:  tipuri_sel.append("Casă")

if not orase_sel or not tipuri_sel:
    st.warning("Selectează cel puțin un oraș și un tip de proprietate.")
else:
    df_cb = df[df["Oraș"].isin(orase_sel) & df["Tip"].isin(tipuri_sel)]

    if not arata_outlieri:
        q99 = df_cb["Preț (EUR)"].quantile(0.99)
        df_cb = df_cb[df_cb["Preț (EUR)"] <= q99]

    fig2 = px.box(
        df_cb, x="Oraș", y="Preț (EUR)", color="Tip",
        title=f"Distribuția prețurilor — {', '.join(orase_sel)}",
        color_discrete_sequence=["#2d6a4f", "#52b788"]
    )
    fig2.update_layout(height=340)
    st.plotly_chart(fig2, use_container_width=True)

    if arata_tabel:
        st.dataframe(
            df_cb[["Oraș", "Cartier", "Tip", "Suprafață (mp)", "Nr. camere", "Preț (EUR)"]].head(20),
            use_container_width=True, hide_index=True
        )

# ── Exercițiu 2 ───────────────────────────────────────────────────
st.subheader("Exercițiul 2")
st.markdown("""
Creează **două checkbox-uri**:
1. `"Afișează statistici descriptive"` — dacă este bifat, afișează `df.describe()`
2. `"Afișează distribuția cartierelor"` — dacă este bifat, afișează un bar chart
   cu numărul de anunțuri per cartier folosind `px.bar`
""")

with st.expander("Indiciu"):
    st.markdown("""
    Creează două checkbox-uri independente cu `st.checkbox`.
    Folosește un `if` pentru fiecare — dacă valoarea returnată este `True`,
    afișează conținutul corespunzător. Pentru bar chart, grupează datele
    cu `.value_counts()` și pasează rezultatul lui `px.bar`.
    """)

cod_ex2 = st.text_area(
    "Codul tău:",
    value="# Creează două checkbox-uri cu conținut diferit în fiecare\n",
    height=140,
    key="ex2_code"
)
if st.button("Run", key="ex2_run"):
    try:
        exec(cod_ex2, {"st": st, "pd": pd, "np": np, "df": df, "px": px})
    except Exception as e:
        st.error(f"Eroare: {e}")

# ═════════════════════════════════════════════════════════════════
# 3. st.form
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("3. Formulare — `st.form`")

st.markdown("""
În mod normal, Streamlit **reexecută întregul script** la fiecare modificare a unui widget.
`st.form` **grupează** mai multe widget-uri și reexecută scriptul **doar** când
utilizatorul apasă butonul de submit.

```python
with st.form("numele_formularului"):
    nume   = st.text_input("Nume")
    varsta = st.slider("Vârstă", 18, 80, 30)
    oras   = st.selectbox("Oraș", ["București", "Cluj"])
    trimis = st.form_submit_button("Trimite")

if trimis:
    st.success(f"Salut {nume} din {oras}, ai {varsta} ani!")
```

**Când folosești `st.form`?**
- Când ai **mai multe inputuri corelate** care trebuie procesate împreună
- Când procesarea e **costisitoare** (antrenare model, calcule grele)
  și nu vrei să o declanșezi la fiecare modificare
- Când construiești un **formular clasic** de tip search/filter/submit
""")

st.subheader("Demo: Caută proprietăți cu formular")

with st.form("cautare_proprietati"):
    st.markdown("**Criterii de căutare**")
    col1, col2, col3 = st.columns(3)
    with col1:
        form_oras = st.selectbox("Oraș", ["Ambele", "București", "Cluj-Napoca"])
        form_tip  = st.radio("Tip", ["Ambele", "Apartament", "Casă"])
    with col2:
        form_camere  = st.slider("Nr. camere (minim)", 1, 6, 2)
        form_pret_max = st.number_input(
            "Preț maxim (EUR)", min_value=20000,
            max_value=2000000, value=300000, step=10000
        )
    with col3:
        form_sup_min = st.number_input("Suprafață minimă (mp)", 20, 300, 50)
        form_an_min  = st.number_input("An construcție minim", 1930, 2024, 2000)

    cautat = st.form_submit_button("Caută proprietăți")

if cautat:
    rez = df.copy()
    if form_oras   != "Ambele": rez = rez[rez["Oraș"]       == form_oras]
    if form_tip    != "Ambele": rez = rez[rez["Tip"]        == form_tip]
    rez = rez[
        (rez["Nr. camere"]      >= form_camere)  &
        (rez["Preț (EUR)"]      <= form_pret_max) &
        (rez["Suprafață (mp)"]  >= form_sup_min)  &
        (rez["An construcție"]  >= form_an_min)
    ]
    if len(rez) == 0:
        st.warning("Nicio proprietate nu corespunde criteriilor. Încearcă alte filtre.")
    else:
        st.success(f"**{len(rez)} proprietăți** găsite.")
        st.dataframe(
            rez[["Oraș","Cartier","Tip","Suprafață (mp)","Nr. camere",
                 "An construcție","Preț (EUR)","Preț/mp (EUR)"]].sort_values("Preț (EUR)"),
            use_container_width=True, hide_index=True
        )

# ── Exercițiu 3 ───────────────────────────────────────────────────
st.subheader("Exercițiul 3")
st.markdown("""
Creează un formular cu:
1. Un `st.selectbox` pentru cartier (din `df["Cartier"].unique()`)
2. Un `st.radio` pentru tip proprietate (`"Apartament"` / `"Casă"`)
3. Un buton de submit `"Analizează"`

Când este trimis, afișează cu `st.metric`:
- Numărul de proprietăți găsite
- Prețul minim și maxim din selecție
""")

with st.expander("Indiciu"):
    st.markdown("""
    În interiorul blocului `with st.form(...)`, poți folosi orice widget normal.
    Butonul de submit se creează cu `st.form_submit_button("text")` — nu cu `st.button`.
    Logica de filtrare și afișare se scrie **în afara** blocului `with`, 
    verificând dacă variabila submit este `True`.
    """)

cod_ex3 = st.text_area(
    "Codul tău:",
    value="# Creează un formular cu selectbox, radio și submit\n# Afișează metrici la submit\n",
    height=160,
    key="ex3_code"
)
if st.button("Run", key="ex3_run"):
    try:
        exec(cod_ex3, {"st": st, "pd": pd, "np": np, "df": df, "px": px})
    except Exception as e:
        st.error(f"Eroare: {e}")

# ═════════════════════════════════════════════════════════════════
# 4. st.download_button
# ═════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("4. Descărcare fișiere — `st.download_button`")

st.markdown("""
`st.download_button` permite utilizatorului să descarce date direct din aplicație —
CSV, Excel, JSON sau orice alt format.

```python
# Descărcare CSV
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Descarcă CSV",
    data=csv,
    file_name="date_export.csv",
    mime="text/csv"
)
```

**Tipuri MIME frecvente:**
| Format  | mime                                                      |
|---------|-----------------------------------------------------------|
| CSV     | `"text/csv"`                                              |
| JSON    | `"application/json"`                                      |
| Excel   | `"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"` |
| Text    | `"text/plain"`                                            |
""")

st.subheader("Demo: Export date filtrate")

col1, col2 = st.columns([3, 1])
with col1:
    oras_export = st.selectbox(
        "Alege orașul pentru export",
        ["Ambele", "București", "Cluj-Napoca"],
        key="export_oras"
    )
with col2:
    tip_export = st.radio(
        "Tip",
        ["Ambele", "Apartament", "Casă"],
        key="export_tip"
    )

df_export = df.copy()
if oras_export != "Ambele": df_export = df_export[df_export["Oraș"] == oras_export]
if tip_export  != "Ambele": df_export = df_export[df_export["Tip"]  == tip_export]

st.write(f"**{len(df_export)} înregistrări** selectate pentru export.")

col_btn1, col_btn2 = st.columns(2)

with col_btn1:
    csv_data = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Descarcă CSV ({len(df_export)} rânduri)",
        data=csv_data,
        file_name=f"imobiliare_{oras_export.lower().replace(' ', '_')}.csv",
        mime="text/csv",
        key="dl_csv"
    )

with col_btn2:
    json_data = df_export.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")
    st.download_button(
        label=f"Descarcă JSON ({len(df_export)} rânduri)",
        data=json_data,
        file_name=f"imobiliare_{oras_export.lower().replace(' ', '_')}.json",
        mime="application/json",
        key="dl_json"
    )

# ── Exercițiu 4 ───────────────────────────────────────────────────
st.subheader("Exercițiul 4")
st.markdown("""
Creează un `st.download_button` care descarcă **doar proprietățile cu valori lipsă**
din dataset, salvate ca `valori_lipsa.csv`.
""")

with st.expander("Indiciu"):
    st.markdown("""
    `df[df.isnull().any(axis=1)]` returnează rândurile care au cel puțin
    o valoare lipsă pe orice coloană. Convertește DataFrame-ul filtrat în CSV
    cu `.to_csv(index=False).encode("utf-8")` și pasează rezultatul
    parametrului `data` al lui `st.download_button`.
    """)

cod_ex4 = st.text_area(
    "Codul tău:",
    value="# Filtrează rândurile cu valori lipsă și creează un download button\n",
    height=130,
    key="ex4_code"
)
if st.button("Run", key="ex4_run"):
    try:
        exec(cod_ex4, {"st": st, "pd": pd, "np": np, "df": df})
    except Exception as e:
        st.error(f"Eroare: {e}")

# ── Rezumat ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Rezumat")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Ce am învățat:**
    - `st.radio()` — selecție exclusivă, vizibilă, orizontală sau verticală
    - `st.checkbox()` — filtre on/off independente, cumulative
    - `st.form()` — grupare inputuri, submit explicit, fără reruns intermediare
    - `st.download_button()` — export CSV, JSON sau orice format
    """)
with col2:
    st.markdown("""
    **Când să folosești ce:**
    - `st.radio` vs `st.selectbox` → puține opțiuni vs multe opțiuni
    - `st.checkbox` → filtre opționale, toggle vizibilitate
    - `st.form` → inputuri corelate, procesare costisitoare
    - `st.download_button` → oricând utilizatorul trebuie să salveze date
    """)

st.info("Continuă cu **Pagina 2 — Preprocesare Interactivă**.")