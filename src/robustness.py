"""Étape 8.3 : test de robustesse synthétique sur la VALIDATION (B vs C).

Les dégradations sont artificielles : indicateur approximatif, pas une mesure
sur de vraies photos de webcam.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import f1_score

from data_utils import get_transforms, load_splits
from predict import get_device, load_trained_model

SEED = 42


def clutter(img, rng):
    """Place l'objet (réduit) sur un fond coloré et irrégulier."""
    w, h = img.size
    noise = rng.integers(0, 256, size=(6, 8, 3), dtype=np.uint8)
    background = Image.fromarray(noise).resize((w, h), Image.Resampling.BICUBIC)
    small = img.resize((int(w * 0.6), int(h * 0.6)))
    x = int(rng.integers(0, w - small.width + 1))
    y = int(rng.integers(0, h - small.height + 1))
    background.paste(small, (x, y))
    return background


CONDITIONS = {
    "original": lambda img, rng: img,
    "sombre": lambda img, rng: ImageEnhance.Brightness(img).enhance(0.35),
    "surexpose": lambda img, rng: ImageEnhance.Brightness(img).enhance(1.8),
    "faible_contraste": lambda img, rng: ImageEnhance.Contrast(img).enhance(0.4),
    "flou": lambda img, rng: img.filter(ImageFilter.GaussianBlur(radius=3)),
    "fond_encombre": clutter,
}


def ece(conf, correct, bins=10):
    """Écart moyen entre confiance affichée et fiabilité réelle."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return total


@torch.no_grad()
def evaluate(model, val_df, condition, device):
    # Même graine pour chaque modèle : les images dégradées sont identiques
    rng = np.random.default_rng(SEED)
    transform = get_transforms(train=False)
    degrade = CONDITIONS[condition]
    probs = []
    for path in val_df["path"]:
        img = degrade(Image.open(path).convert("RGB"), rng)
        x = transform(img).unsqueeze(0).to(device)
        probs.append(torch.softmax(model(x), dim=1)[0].cpu().numpy())
    probs = np.array(probs)

    y_true = val_df["label_idx"].values
    y_pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = y_pred == y_true
    return {
        "condition": condition,
        "accuracy": correct.mean(),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "conf_moyenne": conf.mean(),
        "conf_si_faux": conf[~correct].mean() if (~correct).any() else float("nan"),
        "ece": ece(conf, correct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs")
    parser.add_argument("--tags", nargs="+", default=["finetune", "C"])
    parser.add_argument("--output", default="results/robustness.csv")
    args = parser.parse_args()

    device = get_device()
    _, val_df, _ = load_splits()
    print(f"Validation : {len(val_df)} images | Device : {device}\n")

    rows = []
    for tag in args.tags:
        model, _ = load_trained_model(Path(args.checkpoint_dir) / f"best_{tag}.pth", device)
        for condition in CONDITIONS:
            rows.append({"modele": tag, **evaluate(model, val_df, condition, device)})
        print(f"Modèle {tag} évalué")

    df = pd.DataFrame(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print()
    print(df.round(3).to_string(index=False))
    print(f"\nRésultats sauvegardés : {args.output}")


if __name__ == "__main__":
    main()