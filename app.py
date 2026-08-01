import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

import tensorflow as tf
from keras.models import load_model

# ============================================================
# CONFIGURATION GÉNÉRALE
# ============================================================
st.set_page_config(
    page_title="EyeCare AI",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "EyeCare_AI_VGG16_Final.keras"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

CLASS_META = {
    "cataract": {"label": "Cataracte", "color": "#F4A261", "icon": "🌫️",
                 "desc": "Opacification du cristallin, souvent liée à l'âge."},
    "diabetic_retinopathy": {"label": "Rétinopathie diabétique", "color": "#E76F51", "icon": "🩸",
                              "desc": "Atteinte de la rétine liée au diabète."},
    "glaucoma": {"label": "Glaucome", "color": "#9D4EDD", "icon": "⚠️",
                 "desc": "Atteinte du nerf optique, souvent silencieuse."},
    "normal": {"label": "Normal", "color": "#2A9D8F", "icon": "✅",
               "desc": "Aucune anomalie détectée par le modèle."},
}

# Performances mesurées lors de l'entraînement (issues du classification_report)
MODEL_METRICS = {
    "cataract":              {"precision": 0.89, "recall": 0.88, "f1": 0.88, "support": 204},
    "diabetic_retinopathy":  {"precision": 0.91, "recall": 0.68, "f1": 0.78, "support": 217},
    "glaucoma":              {"precision": 0.79, "recall": 0.72, "f1": 0.76, "support": 201},
    "normal":                {"precision": 0.64, "recall": 0.85, "f1": 0.73, "support": 221},
    
}
GLOBAL_ACCURACY = 0.782

CONF_MATRIX = np.array([
    [180,  0,  17,  7],
    [4, 147,  7, 59],
    [9, 6, 145, 41],
    [10, 9,  14, 188],
])
CONF_LABELS = ["Cataracte", "Rétinopathie diab.", "Glaucome", "Normal"]

# ============================================================
# CSS PERSONNALISÉ
# ============================================================
st.markdown("""
<style>
    .main > div { padding-top: 1.2rem; }

    .hero {
        background: linear-gradient(120deg, #16324F 0%, #2E86AB 100%);
        padding: 2.2rem 2rem;
        border-radius: 18px;
        margin-bottom: 1.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.25);
    }
    .hero h1 { color: white; margin: 0; font-size: 2.2rem; }
    .hero p { color: #DCEFFF; margin-top: 0.4rem; font-size: 1.05rem; }

    .result-card {
        border-radius: 16px;
        padding: 1.6rem;
        text-align: center;
        box-shadow: 0 6px 18px rgba(0,0,0,0.18);
        border: 1px solid rgba(255,255,255,0.08);
    }
    .result-card h2 { margin: 0.3rem 0; font-size: 1.7rem; }
    .result-card .icon { font-size: 2.6rem; }

    .metric-box {
        background: #1A1D24;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        border: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 0.7rem;
    }

    .disclaimer {
        background: rgba(230, 57, 70, 0.12);
        border-left: 4px solid #E63946;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        font-size: 0.92rem;
    }

    .glaucoma-alert {
        background: rgba(157, 78, 221, 0.15);
        border-left: 4px solid #9D4EDD;
        padding: 0.9rem 1.1rem;
        border-radius: 10px;
        font-size: 0.92rem;
        margin-top: 0.8rem;
    }

    section[data-testid="stSidebar"] { border-right: 1px solid rgba(255,255,255,0.06); }

    div[data-testid="stFileUploader"] {
        border: 2px dashed rgba(46, 134, 171, 0.5);
        border-radius: 14px;
        padding: 0.6rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================
@st.cache_resource(show_spinner=False)
def charger_modele():
    return load_model(MODEL_PATH)

def preprocesser_image(image: Image.Image):
    image = image.convert("RGB").resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(image) / 255.0
    return np.expand_dims(array, axis=0)

def predire(model, image: Image.Image):
    array = preprocesser_image(image)
    return model.predict(array, verbose=0)[0]

# ============================================================
# ÉTAT DE SESSION (historique)
# ============================================================
if "history" not in st.session_state:
    st.session_state.history = []

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("## 👁️ EyeCare AI")
    st.caption("Assistant de dépistage — Projet Data Science")

    page = st.radio(
        "Navigation",
        ["🔍 Diagnostic", "📊 Performance du modèle", "🕓 Historique", "ℹ️ À propos"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("**Modèle actif**")
    st.code("VGG16 (Transfer Learning\n+ Fine-tuning)", language=None)
    st.metric("Accuracy globale (validation)", f"{GLOBAL_ACCURACY*100:.1f} %")

    st.divider()
    st.markdown(
        "<div class='disclaimer'>⚠️ Outil pédagogique. Ne remplace pas un avis médical "
        "professionnel.</div>",
        unsafe_allow_html=True,
    )

# ============================================================
# PAGE : DIAGNOSTIC
# ============================================================
if page == "🔍 Diagnostic":

    st.markdown("""
    <div class="hero">
        <h1>👁️ Analyse d'image oculaire</h1>
        <p>Chargez une image de fond d'œil pour obtenir une classification automatique
        parmi 4 catégories : cataracte, rétinopathie diabétique, glaucome, normal.</p>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("Chargement du modèle..."):
        model = charger_modele()

    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded_file = st.file_uploader(
            "Glissez-déposez une image (JPG, JPEG, PNG)",
            type=["jpg", "jpeg", "png"],
        )
    with col_sample:
        st.markdown("&nbsp;")
        st.caption("💡 Astuce : utilisez une image nette, cadrée sur le fond d'œil, sans reflet important.")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        with st.spinner("Analyse en cours..."):
            predictions = predire(model, image)

        predicted_idx = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(predictions[predicted_idx]) * 100
        meta = CLASS_META[predicted_class]

        st.session_state.history.append({
            "Horodatage": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Fichier": uploaded_file.name,
            "Prédiction": meta["label"],
            "Confiance (%)": round(confidence, 1),
        })

        st.divider()
        col_img, col_result, col_gauge = st.columns([1.1, 1, 1])

        with col_img:
            st.image(image, caption="Image analysée", use_container_width=True)

        with col_result:
            st.markdown(f"""
            <div class="result-card" style="background: linear-gradient(160deg, {meta['color']}22, {meta['color']}11); border-color: {meta['color']}55;">
                <div class="icon">{meta['icon']}</div>
                <p style="margin:0; color:#B0B0B0;">Résultat</p>
                <h2 style="color:{meta['color']};">{meta['label']}</h2>
                <p style="color:#CCCCCC; font-size:0.9rem;">{meta['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

            if predicted_class == "rétinopathie diabétique ":
                st.markdown("""
                <div class="glaucoma-alert">
                ⚠️ La rétinopathie diabétique est détecté avec une fiabilité plus faible par ce modèle
                (rappel ≈ 68%). Une consultation ophtalmologique est recommandée en cas de doute,
                quel que soit le résultat.
                </div>
                """, unsafe_allow_html=True)

        with col_gauge:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"suffix": " %", "font": {"size": 34}},
                title={"text": "Confiance du modèle", "font": {"size": 14}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "gray"},
                    "bar": {"color": meta["color"]},
                    "steps": [
                        {"range": [0, 50], "color": "rgba(255,255,255,0.05)"},
                        {"range": [50, 80], "color": "rgba(255,255,255,0.08)"},
                        {"range": [80, 100], "color": "rgba(255,255,255,0.12)"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=230, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig_gauge, use_container_width=True)

        st.markdown("#### Répartition des probabilités par classe")
        proba_df = pd.DataFrame({
            "Classe": [CLASS_META[c]["label"] for c in CLASS_NAMES],
            "Probabilité (%)": [float(p) * 100 for p in predictions],
        }).sort_values("Probabilité (%)", ascending=True)

        fig_bar = px.bar(
            proba_df, x="Probabilité (%)", y="Classe", orientation="h",
            text="Probabilité (%)", color="Classe",
            color_discrete_map={CLASS_META[c]["label"]: CLASS_META[c]["color"] for c in CLASS_NAMES},
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(
            showlegend=False, height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_range=[0, 105],
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Rapport téléchargeable
        report_txt = (
            f"EyeCare AI — Rapport d'analyse\n"
            f"Date : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Fichier : {uploaded_file.name}\n"
            f"Prédiction : {meta['label']}\n"
            f"Confiance : {confidence:.1f}%\n\n"
            f"Probabilités détaillées :\n" +
            "\n".join([f"- {CLASS_META[c]['label']}: {predictions[i]*100:.1f}%" for i, c in enumerate(CLASS_NAMES)]) +
            "\n\nAvertissement : outil pédagogique, ne remplace pas un diagnostic médical."
        )
        st.download_button(
            "⬇️ Télécharger le rapport (.txt)",
            data=report_txt,
            file_name=f"rapport_eyecare_{uploaded_file.name.split('.')[0]}.txt",
            mime="text/plain",
        )

    else:
        st.info("👆 Chargez une image pour lancer l'analyse.")

# ============================================================
# PAGE : PERFORMANCE DU MODÈLE
# ============================================================
elif page == "📊 Performance du modèle":

    st.markdown("""
    <div class="hero">
        <h1>📊 Performance du modèle</h1>
        <p>Métriques mesurées sur le jeu de validation (843 images) après fine-tuning.</p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric-box'><h3>{GLOBAL_ACCURACY*100:.1f}%</h3>Accuracy globale</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><h3>843</h3>Images de validation</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><h3>4</h3>Classes détectées</div>", unsafe_allow_html=True)

    st.markdown("#### Métriques par classe")
    metrics_df = pd.DataFrame(MODEL_METRICS).T.reset_index()
    metrics_df.columns = ["Classe", "Précision", "Rappel", "F1-score", "Support"]
    metrics_df["Classe"] = metrics_df["Classe"].map(lambda c: CLASS_META[c]["label"])

    fig_metrics = px.bar(
        metrics_df.melt(id_vars=["Classe", "Support"], value_vars=["Précision", "Rappel", "F1-score"],
                         var_name="Métrique", value_name="Score"),
        x="Classe", y="Score", color="Métrique", barmode="group",
        color_discrete_sequence=["#2E86AB", "#F4A261", "#2A9D8F"],
    )
    fig_metrics.update_layout(height=380, yaxis_range=[0, 1], margin=dict(t=20))
    st.plotly_chart(fig_metrics, use_container_width=True)

    st.warning(

        "⚠️ Le **rappel de la rétinopathie diabétique (68%)** est la métrique la plus faible "
        "de ce modèle : environ 1 cas sur 3 n'est pas détecté."
        "À interpréter avec prudence."
        
    )

    st.markdown("#### Matrice de confusion")
    fig_cm = px.imshow(
        CONF_MATRIX,
        x=CONF_LABELS, y=CONF_LABELS,
        color_continuous_scale="Blues",
        text_auto=True,
        labels=dict(x="Classe prédite", y="Classe réelle", color="Nombre"),
    )
    fig_cm.update_layout(height=450, margin=dict(t=20))
    st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("#### Tableau détaillé")
    st.dataframe(metrics_df.set_index("Classe"), use_container_width=True)

# ============================================================
# PAGE : HISTORIQUE
# ============================================================
elif page == "🕓 Historique":

    st.markdown("""
    <div class="hero">
        <h1>🕓 Historique des analyses</h1>
        <p>Retrouvez ici les images analysées durant cette session.</p>
    </div>
    """, unsafe_allow_html=True)

    if len(st.session_state.history) == 0:
        st.info("Aucune analyse effectuée pour l'instant. Rendez-vous dans l'onglet Diagnostic.")
    else:
        hist_df = pd.DataFrame(st.session_state.history)
        st.dataframe(hist_df, use_container_width=True, hide_index=True)

        csv = hist_df.to_csv(index=False).encode("utf-8")
        col_a, col_b = st.columns([1, 3])
        with col_a:
            st.download_button("⬇️ Exporter en CSV", csv, "historique_eyecare.csv", "text/csv")
        with col_b:
            if st.button("🗑️ Vider l'historique"):
                st.session_state.history = []
                st.rerun()

        st.markdown("#### Répartition des diagnostics de la session")
        counts = hist_df["Prédiction"].value_counts().reset_index()
        counts.columns = ["Diagnostic", "Nombre"]
        fig_hist = px.pie(counts, names="Diagnostic", values="Nombre", hole=0.5)
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# PAGE : À PROPOS
# ============================================================
elif page == "ℹ️ À propos":

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ À propos du projet</h1>
        <p>Contexte, méthodologie et limites d'EyeCare AI.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    **EyeCare AI** est un projet de fin de formation en Data Science visant à classifier
    automatiquement des images de fond d'œil en 4 catégories : cataracte, rétinopathie
    diabétique, glaucome et œil normal.

    ### 🧠 Méthodologie
    - **Transfer Learning** avec VGG16 pré-entraîné sur ImageNet
    - **Fine-tuning** ciblé des dernières couches convolutives
    - **Data augmentation** (rotation, zoom, translation, flip) pour compenser le volume
      limité de données médicales
    - Diagnostic des erreurs par classe (matrice de confusion, classification report)

    ### 📁 Données
    Dataset "Eye Diseases Classification" (Kaggle), 4 217 images réparties en 4 classes
    équilibrées (~1 000 images/classe).

    ### 📈 Résultat
    **78,2% d'accuracy** sur le jeu de validation — un résultat honnête compte tenu du
    volume de données disponible, avec une faiblesse identifiée sur la détection du glaucome.
    """)

    st.markdown("""
    <div class="disclaimer">
    ⚠️ <b>Avertissement médical important</b><br>
    Cette application est un projet pédagogique de démonstration technique. Elle ne
    constitue en aucun cas un dispositif médical, un outil de diagnostic, ni un substitut
    à une consultation avec un professionnel de santé. Toute décision médicale doit être
    prise avec un médecin ophtalmologiste.
    </div>
    """, unsafe_allow_html=True)

    st.caption("Stack technique : TensorFlow / Keras · Streamlit · Plotly · Python")