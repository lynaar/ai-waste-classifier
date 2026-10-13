"""Étape 5 : inférence sur une seule image."""

import argparse
from pathlib import Path

import torch
from PIL import Image, ImageOps

from data_utils import get_transforms
from model import build_model

DEFAULT_CHECKPOINT = "models/best_finetune.pth"


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_trained_model(checkpoint_path, device=None):
    """Charge le modèle entraîné. Retourne (modèle, liste des classes)."""
    device = device or get_device()
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(pretrained=False)  # architecture seule, poids du checkpoint
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt["classes"]


def prepare_image(image) -> Image.Image:
    """Accepte un chemin ou une image PIL et retourne une image RGB."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    image = ImageOps.exif_transpose(image)  # corrige l'orientation des photos de téléphone
    return image.convert("RGB")  # gère PNG avec transparence, niveaux de gris, etc.


@torch.no_grad()
def predict_image(model, classes, image, device=None) -> dict:
    """Prédit la classe d'une image (chemin ou PIL.Image)."""
    device = device or next(model.parameters()).device
    image = prepare_image(image)
    tensor = get_transforms(train=False)(image).unsqueeze(0).to(device)  # (1, 3, 224, 224)
    probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    best = int(probs.argmax())
    return {
        "label": classes[best],
        "confidence": float(probs[best]),
        "probabilities": {name: float(p) for name, p in zip(classes, probs)},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Prédit la catégorie de déchet d'une image.")
    parser.add_argument("--image", required=True, help="chemin de l'image (JPG, JPEG, PNG)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    model, classes = load_trained_model(args.checkpoint)
    result = predict_image(model, classes, args.image)

    print(f"Image : {args.image}")
    print(f"Prédiction : {result['label']} (confiance : {result['confidence']:.1%})\n")
    print("Probabilités :")
    for name, p in sorted(result["probabilities"].items(), key=lambda kv: -kv[1]):
        print(f"  {name:<10} {p:6.1%}  {'█' * round(p * 30)}")


if __name__ == "__main__":
    main()