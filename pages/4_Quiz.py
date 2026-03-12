import streamlit as st
import pandas as pd

st.title("Quiz — Seminar 4")
st.markdown("Testează ce ai învățat. 10 întrebări, 4 teme.")
st.markdown("---")

# ── Întrebările ───────────────────────────────────────────────────
intrebari = [
    # TEMA 1 — Interactivitate
    {
        "tema": "Interactivitate",
        "intrebare": "Care este diferența principală dintre `st.radio` și `st.selectbox`?",
        "optiuni": [
            "Nu există diferență — fac același lucru",
            "st.radio afișează toate opțiunile simultan; st.selectbox le ascunde într-un dropdown",
            "st.selectbox permite selecție multiplă; st.radio nu",
            "st.radio funcționează doar în sidebar"
        ],
        "corect": 1,
        "explicatie": "`st.radio` afișează toate opțiunile vizibil — potrivit pentru 2-4 opțiuni. "
                      "`st.selectbox` ascunde opțiunile într-un dropdown — potrivit când ai multe opțiuni sau spațiu limitat."
    },
    {
        "tema": "Interactivitate",
        "intrebare": "Ce returnează `st.checkbox('Afișează date')`?",
        "optiuni": [
            "Un string cu textul bifei",
            "Numărul de bifare",
            "True dacă e bifat, False dacă nu",
            "None întotdeauna"
        ],
        "corect": 2,
        "explicatie": "`st.checkbox` returnează o valoare booleană: `True` când este bifat, `False` când nu. "
                      "Se folosește de obicei într-un `if` pentru a controla vizibilitatea unui element."
    },
    {
        "tema": "Interactivitate",
        "intrebare": "De ce folosim `st.form` în loc să punem widget-urile direct în pagină?",
        "optiuni": [
            "Pentru că st.form este mai rapid decât widget-urile normale",
            "Pentru a grupa mai multe inputuri și a declanșa procesarea doar la apăsarea submit",
            "Pentru că widget-urile din afara unui form nu funcționează",
            "st.form este obligatoriu când folosim st.spinner"
        ],
        "corect": 1,
        "explicatie": "Fiecare widget normal declanșează o reexecutare a scriptului la modificare. "
                      "`st.form` grupează inputurile — scriptul se reexecută doar când utilizatorul apasă "
                      "butonul de submit. Util când procesarea este costisitoare (ex: antrenare model ML)."
    },
    # TEMA 2 — Preprocesare
    {
        "tema": "Preprocesare",
        "intrebare": "În dataset-ul nostru, valorile lipsă din coloana `Etaj` sunt:",
        "optiuni": [
            "Erori de colectare a datelor care trebuie corectate",
            "Informații nedisponibile, similar cu `An construcție`",
            "Valori structurale — casele nu au etaj, deci lipsa este intenționată",
            "Duplicate care trebuie eliminate"
        ],
        "corect": 2,
        "explicatie": "Valorile lipsă din `Etaj` sunt structurale — casele nu au etaj, deci absența valorii "
                      "nu este o eroare, ci o caracteristică a proprietății. "
                      "Le tratăm diferit față de `An construcție`, unde lipsa este o informație reală nedisponibilă."
    },
    {
        "tema": "Preprocesare",
        "intrebare": "Care metodă de imputare estimează valorile lipsă pe baza celor mai similare înregistrări din dataset?",
        "optiuni": [
            "Mean imputation",
            "Forward Fill",
            "KNN Imputer",
            "Median imputation"
        ],
        "corect": 2,
        "explicatie": "KNN Imputer (K-Nearest Neighbors) găsește cele mai similare `k` înregistrări "
                      "și estimează valoarea lipsă ca medie a valorilor lor. "
                      "Este mai precis decât Mean sau Median, dar mai lent computațional."
    },
    {
        "tema": "Preprocesare",
        "intrebare": "Când este recomandat să folosești One-Hot Encoding în locul Label Encoding?",
        "optiuni": [
            "Când variabila are o ordine naturală (ex: mic < mediu < mare)",
            "Când variabila este nominală, fără ordine logică între categorii",
            "Când variabila are mai mult de 100 de categorii unice",
            "One-Hot Encoding este întotdeauna mai bun decât Label Encoding"
        ],
        "corect": 1,
        "explicatie": "Label Encoding atribuie numere întregi categoriilor (0, 1, 2...), "
                      "introducând implicit o ordine artificială. "
                      "One-Hot Encoding creează o coloană binară per categorie, fără a implica ordine — "
                      "potrivit pentru variabile nominale ca `Oraș` sau `Cartier`."
    },
    # TEMA 3 — Machine Learning
    {
        "tema": "Machine Learning",
        "intrebare": "Ce înseamnă un R² de 0.85 pentru un model de regresie?",
        "optiuni": [
            "Modelul face erori de 85% în medie",
            "Modelul explică 85% din variația prețurilor din datele de test",
            "Modelul are o acuratețe de 85%",
            "85% din predicții sunt exacte"
        ],
        "corect": 1,
        "explicatie": "R² (coeficientul de determinare) măsoară proporția din varianța variabilei țintă "
                      "explicată de model. R² = 0.85 înseamnă că modelul explică 85% din variația prețurilor. "
                      "R² = 1.0 ar fi perfect; R² = 0.0 înseamnă că modelul nu e mai bun decât media."
    },
    {
        "tema": "Machine Learning",
        "intrebare": "Care este scopul împărțirii datelor în set de antrenare și set de testare?",
        "optiuni": [
            "Pentru a reduce timpul de antrenare",
            "Pentru a evalua cât de bine generalizează modelul pe date nevăzute",
            "Pentru că scikit-learn nu funcționează fără această împărțire",
            "Pentru a crește acuratețea modelului"
        ],
        "corect": 1,
        "explicatie": "Dacă evaluăm modelul pe aceleași date pe care l-am antrenat, obținem rezultate "
                      "artificial bune — modelul a 'memorat' datele. "
                      "Setul de testare simulează date noi, nevăzute, și măsoară cât de bine "
                      "generalizează modelul în practică."
    },
    # TEMA 4 — Impact preprocesare
    {
        "tema": "Impact preprocesare",
        "intrebare": "De ce poate scădea performanța modelului dacă eliminăm outlierii înainte de antrenare?",
        "optiuni": [
            "Outlierii nu afectează niciodată modelul — e un mit",
            "Eliminând outlieri reducem setul de date, ceea ce crește overfitting-ul",
            "Outlierii pot conține informație reală — ex: proprietăți premium pe care modelul trebuie să le învețe",
            "scikit-learn nu funcționează corect fără outlieri"
        ],
        "corect": 2,
        "explicatie": "În dataset-ul nostru, outlierii sunt în majoritate proprietăți de lux cu prețuri ridicate. "
                      "Dacă îi eliminăm, modelul nu mai are exemple din care să învețe segmentul Premium. "
                      "Decizia de a păstra sau elimina outlierii depinde de contextul problemei."
    },
    {
        "tema": "Impact preprocesare",
        "intrebare": "Care dintre următoarele afirmații descrie cel mai bine relația dintre preprocesare și performanța modelului?",
        "optiuni": [
            "Preprocesarea nu afectează performanța — doar algoritmul contează",
            "Mai multă preprocesare duce întotdeauna la rezultate mai bune",
            "Alegerile de preprocesare influențează semnificativ performanța și trebuie validate experimental",
            "Preprocesarea contează doar pentru modele simple; Random Forest este robust la orice date"
        ],
        "corect": 2,
        "explicatie": "Am demonstrat în secțiunea de comparație că același model Random Forest "
                      "poate obține R² între 0.3 și 0.9 în funcție de alegerile de preprocesare. "
                      "Nu există o rețetă universală — fiecare decizie trebuie testată și justificată."
    },
]

# ── Inițializare session_state ────────────────────────────────────
if "quiz_raspunsuri" not in st.session_state:
    st.session_state.quiz_raspunsuri = {}
if "quiz_verificat" not in st.session_state:
    st.session_state.quiz_verificat = False

# ── Afișare întrebări ─────────────────────────────────────────────
tema_curenta = None

for i, q in enumerate(intrebari):
    # Header temă
    if q["tema"] != tema_curenta:
        tema_curenta = q["tema"]
        st.markdown(f"### {tema_curenta}")

    raspuns = st.radio(
        f"**{i+1}. {q['intrebare']}**",
        options=q["optiuni"],
        index=None,
        key=f"q_{i}"
    )
    st.session_state.quiz_raspunsuri[i] = raspuns

    # Feedback după verificare
    if st.session_state.quiz_verificat and raspuns is not None:
        idx_ales = q["optiuni"].index(raspuns)
        if idx_ales == q["corect"]:
            st.success(f"Corect! {q['explicatie']}")
        else:
            st.error(
                f"Incorect. Răspunsul corect: **{q['optiuni'][q['corect']]}**\n\n{q['explicatie']}"
            )

    st.markdown("")

# ── Butoane ───────────────────────────────────────────────────────
st.markdown("---")
col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Verifică răspunsurile", type="primary", key="verifica"):
        neselectate = [
            i + 1 for i, r in st.session_state.quiz_raspunsuri.items()
            if r is None
        ]
        if neselectate:
            st.warning(
                f"Ai {len(neselectate)} întrebări fără răspuns: "
                f"{', '.join(map(str, neselectate))}. Completează-le înainte de verificare."
            )
        else:
            st.session_state.quiz_verificat = True
            st.rerun()

with col2:
    if st.button("Resetează quiz", key="reset"):
        st.session_state.quiz_raspunsuri = {}
        st.session_state.quiz_verificat = False
        st.rerun()

# ── Scor ──────────────────────────────────────────────────────────
if st.session_state.quiz_verificat:
    st.markdown("---")

    raspunsuri = st.session_state.quiz_raspunsuri
    corecte = sum(
        1 for i, q in enumerate(intrebari)
        if raspunsuri.get(i) is not None
        and q["optiuni"].index(raspunsuri[i]) == q["corect"]
    )
    total = len(intrebari)
    procent = corecte / total

    # Scor general
    col1, col2, col3 = st.columns(3)
    col1.metric("Scor", f"{corecte} / {total}")
    col2.metric("Procent", f"{procent:.0%}")
    if procent == 1.0:
        col3.metric("Rezultat", "Perfect!")
    elif procent >= 0.7:
        col3.metric("Rezultat", "Bine!")
    else:
        col3.metric("Rezultat", "Mai încearcă")

    # Scor per temă
    st.markdown("#### Rezultate per temă")
    teme = ["Interactivitate", "Preprocesare", "Machine Learning", "Impact preprocesare"]
    cols = st.columns(4)

    for j, tema in enumerate(teme):
        q_tema = [q for q in intrebari if q["tema"] == tema]
        corecte_tema = sum(
            1 for q in q_tema
            if raspunsuri.get(intrebari.index(q)) is not None
            and q["optiuni"].index(raspunsuri[intrebari.index(q)]) == q["corect"]
        )
        cols[j].metric(tema, f"{corecte_tema} / {len(q_tema)}")