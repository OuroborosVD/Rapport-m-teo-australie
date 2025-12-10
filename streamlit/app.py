import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import joblib

# CONFIG GÉNÉRALE


st.set_page_config(
    page_title="Prévision météo",
    page_icon="🌦️",
    layout="wide"
)


# STYLES PERSONNALISÉS (CSS)


st.markdown(
    """
    <style>
    .main { background-color: #F7FAFC; }
    .title {
        color: #2B6CB0;
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #38B2AC;
        font-size: 20px;
        font-weight: 500;
        margin-bottom: 1.5rem;
    }
    .section-title {
        background-color: #EDF2F7;
        padding: 8px 14px;
        border-radius: 8px;
        font-size: 18px;
        font-weight: 700;
        color: #2D3748;
        border-left: 5px solid #2B6CB0;
        margin-top: 25px;
        margin-bottom: 10px;
    }
    .info-box {
        background-color: #FFFFFF;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(160, 174, 192, 0.3);
        margin-bottom: 10px;
    }
    .result-ok {
        background-color: #C6F6D5;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #38A169;
        margin-top: 15px;
    }
    .result-rain {
        background-color: #BEE3F8;
        padding: 12px 16px;
        border-radius: 10px;
        border: 1px solid #3182CE;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# CONSTANTES


MODEL_PATH = "models/hgb_model.joblib"  # <-- À créer plus tard avec votre vrai modèle
THRESHOLD = 0.69  # seuil décisionnel



# CHARGEMENT DES DONNÉES


@st.cache_data
def load_data(path: str):
    file_path = Path(path)
    if file_path.exists():
        return pd.read_csv(file_path)
    return None

df = load_data("data/weatherAUS_encoded.csv")



# CHARGEMENT DU MODÈLE


@st.cache_resource
def load_model(path: str):
    file_path = Path(path)
    if file_path.exists():
        try:
            return joblib.load(file_path)
        except Exception:
            return None
    return None

model = load_model(MODEL_PATH)

# Base d'entrée pour la prédiction : on part d'une ligne du dataset
base_input = None
if df is not None:
    feature_cols = [c for c in df.columns if c != "RainTomorrow"]
    if len(feature_cols) > 0:
        base_input = df[feature_cols].iloc[0:1].copy()



# MENU SIDEBAR


st.sidebar.title("🌦️ Projet météo Australie")
page = st.sidebar.radio(
    "Navigation",
    ["Accueil", "Exploration des données", "Modélisation", "Prédiction"]
)
st.sidebar.markdown("---")
st.sidebar.write("Projet DataScientest – Iris, Jean-Paul & Louis")



# PAGE : ACCUEIL

if page == "Accueil":

    st.markdown("<div class='title'>🌦️ Projet météo Australie</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Analyse et prédiction de la pluie à partir des données WeatherAUS</div>", unsafe_allow_html=True)

    st.markdown("Projet réalisé par **Iris**, **Jean-Paul** et **Louis** dans le cadre de la formation DataScientest.")

    st.markdown("<div class='section-title'>🎯 Objectif du projet</div>", unsafe_allow_html=True)
    st.markdown(
        """
        L'objectif est de **prédire s'il pleuvra le lendemain (RainTomorrow)** en Australie.

        Le projet comprend :
        - nettoyage et préparation des données  
        - exploration et visualisation  
        - entraînement de plusieurs modèles ML  
        - optimisation via RandomizedSearch  
        - comparaison des performances  
        - sélection d’un meilleur modèle  
        - intégration dans une application Streamlit  
        """
    )

    st.markdown("<div class='section-title'>📊 Informations sur le dataset</div>", unsafe_allow_html=True)
    if df is not None:
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.write(f"**Nombre de lignes :** {len(df):,}".replace(",", " "))
        st.write(f"**Nombre de colonnes :** {df.shape[1]}")
        if "RainTomorrow" in df.columns:
            st.write(f"**Taux de jours de pluie :** {df['RainTomorrow'].mean()*100:.1f}%")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.error("Impossible de charger le fichier weatherAUS_encoded.csv")



# PAGE : EXPLORATION


elif page == "Exploration des données":

    st.markdown("<div class='title'>🔍 Exploration des données</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Aperçu visuel du dataset WeatherAUS</div>", unsafe_allow_html=True)

    if df is None:
        st.error("❌ Fichier introuvable.")
    else:
        st.success("✅ Données chargées")

        st.markdown("<div class='section-title'>📊 Informations générales</div>", unsafe_allow_html=True)
        st.markdown("<div class='info-box'>", unsafe_allow_html=True)
        st.write(f"**Lignes :** {len(df):,}".replace(",", " "))
        st.write(f"**Colonnes :** {df.shape[1]}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section-title'>👀 Aperçu des données</div>", unsafe_allow_html=True)
        st.dataframe(df.head(10))

        # Graphique RainTomorrow avec 2 couleurs
        if "RainTomorrow" in df.columns:

            st.markdown("<div class='section-title'>🌧️ Répartition de RainTomorrow</div>", unsafe_allow_html=True)

            ct = df["RainTomorrow"].value_counts().sort_index()
            ct_df = ct.reset_index()
            ct_df.columns = ["RainTomorrow", "Nombre"]
            ct_df["RainTomorrow"] = ct_df["RainTomorrow"].astype(str)

            chart = (
                alt.Chart(ct_df)
                .mark_bar()
                .encode(
                    x=alt.X("RainTomorrow:N", title="RainTomorrow (0 = pas de pluie, 1 = pluie)"),
                    y=alt.Y("Nombre:Q", title="Nombre d'observations"),
                    color=alt.Color(
                        "RainTomorrow:N",
                        scale=alt.Scale(range=["#4A90E2", "#FF6F91"]),
                        legend=None
                    ),
                    tooltip=["RainTomorrow", "Nombre"]
                )
                .properties(width=500, height=350)
            )

            st.altair_chart(chart, use_container_width=True)

        else:
            st.info("La colonne RainTomorrow n'existe pas.")



# PAGE : MODÉLISATION


elif page == "Modélisation":

    st.markdown("<div class='title'>🤖 Modélisation</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Comparaison des modèles de Machine Learning</div>", unsafe_allow_html=True)

    st.markdown(
        """
        Trois modèles ont été entraînés :

        - **RandomForestClassifier**  
        - **GradientBoostingClassifier**  
        - **HistGradientBoostingClassifier** (meilleur modèle)

        Les métriques analysées :
        - F1-score (classe pluie)  
        - Précision  
        - Rappel  
        - Accuracy globale  
        """
    )

    st.markdown("<div class='section-title'>📊 Résultats des modèles</div>", unsafe_allow_html=True)

    data_scores = {
        "Modèle": [
            "RandomForest",
            "GradientBoosting",
            "HistGradientBoosting (meilleur)"
        ],
        "F1 pluie": [0.63, 0.63, 0.69],
        "Précision pluie": [0.68, 0.70, 0.71],
        "Rappel pluie": [0.59, 0.58, 0.67],
        "Accuracy globale": [0.84, 0.84, 0.86]
    }

    df_scores = pd.DataFrame(data_scores)

    st.dataframe(
        df_scores.style.format({
            "F1 pluie": "{:.2f}",
            "Précision pluie": "{:.2f}",
            "Rappel pluie": "{:.2f}",
            "Accuracy globale": "{:.2f}"
        })
    )

    st.markdown("<div class='section-title'>🥇 Meilleur modèle</div>", unsafe_allow_html=True)

    st.markdown(
        """
        Le modèle **HistGradientBoostingClassifier** est retenu car :

        - meilleur **F1-score** (0.69)  
        - meilleure **accuracy globale** (0.86)  
        - excellent compromis rappel/précision  
        - très performant sur les grands jeux de données  

        👉 Ce modèle sera utilisé dans la page **Prédiction**.
        """
    )

    st.markdown(
        "<div class='info-box'>🌈 Modèle final : HistGradientBoosting (seuil optimisé 0.69)</div>",
        unsafe_allow_html=True
    )



# PAGE : PRÉDICTION


elif page == "Prédiction":

    st.markdown("<div class='title'>🌈 Prédiction de la pluie</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Utilisation du modèle final pour prédire RainTomorrow</div>", unsafe_allow_html=True)

    if df is None or base_input is None:
        st.error("Les données ne sont pas disponibles, impossible de préparer la prédiction.")
    else:
        if model is None:
            st.warning(
                "⚠️ Le fichier de modèle n'a pas été trouvé.\n\n"
                f"Placez votre modèle entraîné dans **`{MODEL_PATH}`** "
                "pour activer la prédiction automatique."
            )

        st.markdown("<div class='section-title'>📝 Saisir les informations météo</div>", unsafe_allow_html=True)

        with st.form("prediction_form"):
            cols = st.columns(2)

            user_values = {}

            numeric_features = [
                ("MinTemp", "Température minimale (°C)"),
                ("MaxTemp", "Température maximale (°C)"),
                ("Rainfall", "Volume de pluie aujourd'hui (mm)"),
                ("Humidity3pm", "Humidité à 15h (%)"),
                ("WindSpeed3pm", "Vitesse du vent à 15h (km/h)"),
                ("Pressure3pm", "Pression à 15h (hPa)"),
                ("Temp3pm", "Température à 15h (°C)")
            ]

            for i, (col, label) in enumerate(numeric_features):
                if col in df.columns:
                    default_val = float(df[col].median())
                    with cols[i % 2]:
                        user_values[col] = st.number_input(
                            label,
                            value=default_val,
                            step=0.1
                        )

            if "RainToday" in df.columns:
                with cols[0]:
                    rain_today = st.selectbox(
                        "Pluie aujourd'hui ?",
                        options=[0, 1],
                        format_func=lambda x: "Non" if x == 0 else "Oui"
                    )
                    user_values["RainToday"] = rain_today

            submitted = st.form_submit_button("Lancer la prédiction")

        if submitted:
            input_row = base_input.copy()

            for col, val in user_values.items():
                if col in input_row.columns:
                    input_row[col] = val

            if model is None:
                st.info(
                    "✅ Les données d'entrée sont prêtes, mais le modèle n'est pas encore connecté.\n\n"
                    f"Enregistrez votre modèle dans **`{MODEL_PATH}`** pour obtenir une prédiction."
                )
            else:
                try:
                    proba = model.predict_proba(input_row)[0, 1]
                    prediction = int(proba >= THRESHOLD)

                    st.markdown("<div class='section-title'>📌 Résultat</div>", unsafe_allow_html=True)
                    st.write(f"**Probabilité de pluie demain : {proba*100:.1f} %** (seuil = {THRESHOLD})")

                    if prediction == 1:
                        st.markdown(
                            "<div class='result-rain'>🌧️ Le modèle prédit qu'il **pleuvra** demain.</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            "<div class='result-ok'>☀️ Le modèle prédit qu'il ne **pleuvra pas** demain.</div>",
                            unsafe_allow_html=True
                        )

                except Exception as e:
                    st.error(
                        "Une erreur est survenue lors de la prédiction. "
                        "Vérifiez que le modèle est compatible avec les colonnes du dataset."
                    )
import streamlit as st

st.subheader("📊 Comparaison des modèles (résultats d'équipe)")

st.markdown("""
| Modèle                     | F1 pluie | Précision pluie | Rappel pluie | Accuracy |
|---------------------------|---------:|----------------:|-------------:|---------:|
| Régression logistique     | 0.60     | 0.xx            | 0.xx         | 0.84     |
| RandomForest optimisé | 0.63  | 0.68           | 0.59         | 0.84     |
| Gradient Boosting         | 0.63     | 0.70            | 0.58         | 0.84     |
| HistGradientBoosting (final) | 0.69  | 0.71           | 0.67         | 0.86     |
""")



