import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Imobiliare România — Seminar 4",
    page_icon="🏠",
    layout="wide"
)

# ── Dataset integrat — nu necesită încărcare ──────────────────────
@st.cache_data
def load_data():
    return pd.read_csv("imobiliare_romania.csv")

df = load_data()

# ── Sidebar ───────────────────────────────────────────────────────
st.sidebar.title("Seminar 4")
st.sidebar.markdown("**Streamlit & Machine Learning**")
st.sidebar.markdown("---")
st.sidebar.markdown(f"**Dataset:** {len(df)} înregistrări")
st.sidebar.markdown(f"**Orașe:** {', '.join(df['Oraș'].unique())}")

# ── Hero ──────────────────────────────────────────────────────────
st.title("Piața Imobiliară din România")
st.markdown(
    "#### Un dashboard complet — de la date brute la modele de Machine Learning"
)
st.markdown("---")

# ── Introducere ───────────────────────────────────────────────────
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("""
    ### Ce vom construi astăzi

    Pornind de la un dataset real de anunțuri imobiliare din **București** și
    **Cluj-Napoca**, vom parcurge împreună întregul flux al unui proiect de date:

    1. **Interactivitate avansată** — widget-uri noi, exerciții live, pattern-ul
       *Learn → Code → Verify*
    2. **Preprocesare interactivă** — tratarea valorilor lipsă, outlierilor și
       variabilelor categorice. Alegerile tale contează.
    3. **Machine Learning** — două modele, același dataset, preprocesări diferite.
       Vei vedea concret cum deciziile de la pasul 2 influențează performanța modelului.
    4. **Quiz** și **Proiect de Grup**
    """)

with col2:
    st.markdown("### Streamlit Quick Recap")
    st.info("""
    **Concepte deja cunoscute:**
    - Structură multipage cu `pages/`
    - `st.session_state` pentru date persistente
    - `st.dataframe`, `st.metric`, `st.plotly_chart`
    - `st.file_uploader`, filtre în sidebar

    **Concepte noi azi:**
    - `st.radio`, `st.checkbox`, `st.form`
    - `st.download_button`, `st.progress`
    - `st.data_editor`
    - Integrare `scikit-learn` în Streamlit
    """)

st.markdown("---")

# ── Prezentarea paginilor ─────────────────────────────────────────
st.markdown("### Ce vei învăța pe fiecare pagină")

c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    **Pagina 1 — Interactivitate Avansată**

    `st.radio` · `st.checkbox` · `st.form` · `st.download_button`

    Pattern *Learn → Code → Verify*: fiecare widget explicat,
    demonstrat, apoi exersat cu cod pe care îl scrii tu.
    """)
    st.markdown("""
    **Pagina 4 — Quiz**

    10 întrebări din toate temele seminarului.
    Feedback imediat cu explicații.
    """)

with c2:
    st.markdown("""
    **Pagina 2 — Preprocesare Interactivă**

    Valori lipsă · Outlieri · Encoding

    Alegi metoda de imputare — Mean, Median, ffill, KNN.
    Rezultatul se vede instant în date și în statistici.
    """)
    st.markdown("""
    **Pagina 5 — Proiect de Grup**

    30 de minute, dataset la alegere, deploy live
    pe Streamlit Community Cloud.
    """)

with c3:
    st.markdown("""
    **Pagina 3 — Machine Learning**

    Regresie (preț) · Clasificare (segment de preț)

    Același model antrenat cu preprocesări diferite.
    Vei compara R², RMSE și Accuracy side-by-side
    și vei înțelege de ce preprocesarea contează.
    """)

st.markdown("---")

# ── Preview dataset ───────────────────────────────────────────────
st.markdown("### Dataset-ul seminarului")

st.markdown("""
Datele sunt deja încărcate — nu trebuie să faci nimic.
Folosim `@st.cache_data` pentru a evita recitirea fișierului
la fiecare interacțiune.
""")

with st.expander("Cum funcționează @st.cache_data?"):
    st.code("""
@st.cache_data
def load_data():
    return pd.read_csv("imobiliare_romania.csv")

df = load_data()
# Prima dată: citește fișierul și memorează rezultatul
# Următoarele rulări: returnează direct rezultatul din cache
# Avantaj: aplicația nu recitește fișierul la fiecare click
""", language="python")

# Metrici rapide
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Total anunțuri", f"{len(df):,}")
m2.metric("București", f"{(df['Oraș'] == 'București').sum():,}")
m3.metric("Cluj-Napoca", f"{(df['Oraș'] == 'Cluj-Napoca').sum():,}")
m4.metric("Preț mediu", f"{df['Preț (EUR)'].mean():,.0f} €")
m5.metric("Preț/mp mediu", f"{df['Preț/mp (EUR)'].mean():,.0f} €/mp")

st.markdown("<br>", unsafe_allow_html=True)

# Tabs preview
tab_date, tab_structura, tab_lipsa = st.tabs([
    "Primele înregistrări",
    "Structura coloanelor",
    "Valori lipsă"
])

with tab_date:
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

with tab_structura:
    info = pd.DataFrame({
        "Coloană": df.columns,
        "Tip": df.dtypes.astype(str).values,
        "Valori unice": [df[c].nunique() for c in df.columns],
        "Exemplu": [str(df[c].dropna().iloc[0]) if len(df[c].dropna()) > 0 else "—"
                    for c in df.columns]
    })
    st.dataframe(info, use_container_width=True, hide_index=True)

with tab_lipsa:
    lipsa = pd.DataFrame({
        "Coloană": df.columns,
        "Valori lipsă": df.isnull().sum().values,
        "Procent (%)": (df.isnull().sum().values / len(df) * 100).round(1)
    })
    lipsa = lipsa[lipsa["Valori lipsă"] > 0]
    if len(lipsa) > 0:
        st.warning(f"**{len(lipsa)} coloane** conțin valori lipsă — le vom trata pe Pagina 2.")
        st.dataframe(lipsa, use_container_width=True, hide_index=True)
        st.markdown("""
        **De ce există valori lipsă?**
        - `Etaj` — casele nu au etaj (valoare lipsă = casă, nu eroare)
        - `An construcție` — informație nedisponibilă în anunț
        """)
    else:
        st.success("Nicio valoare lipsă!")

st.markdown("---")
st.info("Navighează la **Pagina 1** din sidebar pentru a începe.")