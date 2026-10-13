"""Étape 6 : application Streamlit de classification de déchets."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

# predict.py et data_utils.py sont dans src/
sys.path.append(str(Path(__file__).parent / "src"))
from predict import load_trained_model, predict_image, prepare_image  # noqa: E402

CHECKPOINT = Path(__file__).parent / "models" / "best_finetune.pth"
# Seuil d'avertissement : choix d'interface, PAS une valeur mesurée
LOW_CONFIDENCE = 0.50

st.set_page_config(page_title="AI Waste Classifier", page_icon="♻️", layout="centered")


@st.cache_resource(show_spinner="Chargement du modèle...")
def get_model(checkpoint_path: str):
    """Chargé une seule fois, puis réutilisé à chaque interaction."""
    return load_trained_model(checkpoint_path)


def show_result(image: Image.Image, result: dict) -> None:
    col_image, col_result = st.columns(2)
    with col_image:
        st.image(image, caption="Image analysée")
    with col_result:
        st.subheader(result["label"].capitalize())
        st.metric("Confiance", f"{result['confidence']:.1%}")
        if result["confidence"] < LOW_CONFIDENCE:
            st.warning(
                "Prédiction incertaine. Essayez avec l'objet seul, "
                "sur un fond uni et clair, bien éclairé."
            )

    with st.expander("Probabilités des 6 classes"):
        probs = pd.Series(result["probabilities"]).sort_values(ascending=False)
        st.bar_chart(probs)


def classify(file, model, classes) -> None:
    """Lit le fichier (upload ou webcam), prédit et affiche."""
    try:
        image = Image.open(file)
        image.load()
    except (UnidentifiedImageError, OSError):
        st.error("Impossible de lire ce fichier. Utilisez une image JPG, JPEG ou PNG valide.")
        return
    result = predict_image(model, classes, image)  # même prétraitement qu'à l'évaluation
    show_result(prepare_image(image), result)


def main() -> None:
    st.title("♻️ AI Waste Classifier")
    st.caption(
        "Classe un déchet en six catégories : cardboard, glass, metal, paper, plastic, trash. "
        "ResNet18 (transfert d'apprentissage) entraîné sur TrashNet."
    )

    if not CHECKPOINT.exists():
        st.error(f"Modèle introuvable : `{CHECKPOINT}`. Placez le checkpoint à cet emplacement.")
        st.stop()
    model, classes = get_model(str(CHECKPOINT))

    tab_upload, tab_camera = st.tabs(["📁 Importer une image", "📷 Caméra"])

    with tab_upload:
        uploaded = st.file_uploader("Choisissez une image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            classify(uploaded, model, classes)

    with tab_camera:
        st.info("Autorisez l'accès à la caméra quand le navigateur le demande.")
        photo = st.camera_input("Prenez une photo de l'objet")
        if photo is not None:
            classify(photo, model, classes)
        with st.expander("Problème de caméra ?"):
            st.markdown(
                "- Cliquez sur l'icône de cadenas près de l'adresse et autorisez la **caméra**.\n"
                "- Fermez les autres applications qui utilisent la caméra (Zoom, Teams...).\n"
                "- La caméra exige une connexion **HTTPS** (ou `localhost`).\n"
                "- Sinon, utilisez l'onglet **Importer une image**."
            )

    st.divider()
    st.caption(
        "Limite : le modèle a vu des photos TrashNet (objet seul sur fond clair). "
        "Sur d'autres photos, la précision peut être plus faible. Il choisit toujours "
        "une des six classes, même pour un objet hors catégories."
    )


main()