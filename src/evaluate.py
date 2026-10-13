"""Étape 4 : évaluation des checkpoints sur le test set."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # pas d'affichage : on sauvegarde seulement les figures
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from data_utils import CLASSES, WasteDataset, get_transforms, load_splits
from model import build_model

MODES = ["head", "finetune"]
TITLES = {
    "head": "A : couche finale seule",
    "finetune": "B : fine-tuning de layer4",
    "C": "C : layer3+4, augmentation forte",
    "D": "D : C + objet réduit sur fonds synthétiques",
}


def load_model(path: Path, device):
    """Reconstruit ResNet18 et charge les poids sauvegardés."""
    ckpt = torch.load(path, map_location=device)
    model = build_model(pretrained=False)  # pas besoin de re-télécharger ImageNet
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device), ckpt


@torch.no_grad()
def predict(model, loader, device):
    """Retourne (vrais labels, labels prédits, probabilités)."""
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in loader:
        outputs = model(images.to(device))
        all_probs.append(torch.softmax(outputs, dim=1).cpu().numpy())
        all_labels.extend(labels.tolist())
    probs = np.concatenate(all_probs)
    return np.array(all_labels), probs.argmax(axis=1), probs


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False
    )
    ax.set_title(f"Matrice de confusion (test) - {title}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def plot_examples(test_df, y_true, y_pred, probs, indices, title, path):
    if len(indices) == 0:
        return
    cols = 4
    rows = int(np.ceil(len(indices) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.3))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, indices):
        ax.imshow(Image.open(test_df.loc[i, "path"]).convert("RGB"))
        confidence = probs[i, y_pred[i]]
        ax.set_title(
            f"vrai : {CLASSES[y_true[i]]}\n"
            f"prédit : {CLASSES[y_pred[i]]} ({confidence:.0%})",
            fontsize=9,
        )
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)


def plot_loss_curves(checkpoint_dir: Path, path: Path, modes) -> None:
    fig, axes = plt.subplots(1, len(modes), figsize=(5.5 * len(modes), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, mode in zip(axes, modes):
        history = pd.read_csv(checkpoint_dir / f"history_{mode}.csv")
        ax.plot(history["epoch"], history["train_loss"], label="train")
        ax.plot(history["epoch"], history["val_loss"], label="validation")
        ax.set_title(TITLES[mode], fontsize=10)
        ax.set_xlabel("Époque")
        ax.legend()
    axes[0].set_ylabel("Loss")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-dir", default="outputs")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--modes", nargs="+", default=MODES)
    args = parser.parse_args()

    checkpoint_dir, out_dir = Path(args.checkpoint_dir), Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, _, test_df = load_splits()
    test_ds = WasteDataset(test_df, get_transforms(train=False))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    print(f"Test set : {len(test_df)} images | Device : {device}\n")

    summary = []
    for mode in args.modes:
        model, ckpt = load_model(checkpoint_dir / f"best_{mode}.pth", device)
        y_true, y_pred, probs = predict(model, test_loader, device)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        summary.append({
            "configuration": TITLES[mode],
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "checkpoint_epoch": ckpt["epoch"],
            "val_f1_macro": ckpt["val_macro_f1"],
        })

        # Détail par classe
        report = classification_report(
            y_true, y_pred, target_names=CLASSES, output_dict=True, zero_division=0
        )
        pd.DataFrame(report).T.to_csv(out_dir / f"classification_report_{mode}.csv")
        print(f"=== {TITLES[mode]} ===")
        print(classification_report(
            y_true, y_pred, target_names=CLASSES, digits=3, zero_division=0
        ))

        # Prédictions complètes (utile pour analyser les erreurs)
        pred_df = test_df[["path", "label"]].copy()
        pred_df["predicted"] = [CLASSES[i] for i in y_pred]
        pred_df["confidence"] = probs.max(axis=1)
        pred_df.to_csv(out_dir / f"predictions_{mode}.csv", index=False)

        # Figures
        plot_confusion(y_true, y_pred, TITLES[mode], out_dir / f"confusion_matrix_{mode}.png")

        correct = np.where(y_true == y_pred)[0]
        wrong = np.where(y_true != y_pred)[0]
        rng = np.random.default_rng(42)
        correct_pick = rng.choice(correct, size=min(8, len(correct)), replace=False)
        # Erreurs les plus confiantes en premier
        wrong_sorted = wrong[np.argsort(-probs[wrong, y_pred[wrong]])][:8]
        plot_examples(test_df, y_true, y_pred, probs, correct_pick,
                      f"Exemples bien classés - {TITLES[mode]}",
                      out_dir / f"examples_correct_{mode}.png")
        plot_examples(test_df, y_true, y_pred, probs, wrong_sorted,
                      f"Erreurs (les plus confiantes) - {TITLES[mode]}",
                      out_dir / f"examples_errors_{mode}.png")
        print(f"Erreurs : {len(wrong)} / {len(y_true)}\n")

    plot_loss_curves(checkpoint_dir, out_dir / "loss_curves.png", args.modes)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(out_dir / "metrics_comparison.csv", index=False)
    print("=== Comparaison sur le test set ===")
    print(summary_df.round(4).to_string(index=False))
    print(f"\nRésultats sauvegardés dans : {out_dir}")


if __name__ == "__main__":
    main()