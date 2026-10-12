from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

DATA_DIR = Path("data/dataset-resized")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def collect_images(data_dir: Path) -> pd.DataFrame:
    """Parcourt les dossiers et retourne un DataFrame (chemin, classe)."""
    rows = []
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Dossier manquant : {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": str(path), "label": class_name})
    return pd.DataFrame(rows)


def check_image(path: str):
    """Tente d'ouvrir et de décoder l'image.

    Retourne (largeur, hauteur, mode) si OK, sinon None.
    """
    try:
        with Image.open(path) as img:
            img.verify()  # vérifie l'intégrité du fichier
        with Image.open(path) as img:
            img.load()  # force le décodage complet des pixels
            return img.size[0], img.size[1], img.mode
    except Exception as error:
        print(f"[CORROMPUE] {path} -> {error}")
        return None


def main() -> None:
    df = collect_images(DATA_DIR)
    print(f"Nombre total d'images trouvées : {len(df)}\n")

    # Distribution des classes
    counts = df["label"].value_counts().reindex(CLASSES)
    summary = pd.DataFrame(
        {"images": counts, "pourcentage": (counts / len(df) * 100).round(1)}
    )
    print("Distribution des classes :")
    print(summary, "\n")

    # Vérification des images
    results = [check_image(p) for p in df["path"]]
    df["ok"] = [r is not None for r in results]
    df["width"] = [r[0] if r else None for r in results]
    df["height"] = [r[1] if r else None for r in results]
    df["mode"] = [r[2] if r else None for r in results]

    n_bad = (~df["ok"]).sum()
    print(f"Images corrompues ou illisibles : {n_bad}")
    print("Tailles d'images (largeur x hauteur) :")
    print(df[df["ok"]].groupby(["width", "height"]).size().to_string(), "\n")
    print("Modes de couleur :")
    print(df[df["ok"]]["mode"].value_counts().to_string(), "\n")

    # Graphique
    plt.figure(figsize=(8, 4))
    plt.bar(CLASSES, counts.values)
    plt.title("TrashNet : nombre d'images par classe")
    plt.ylabel("Nombre d'images")
    plt.tight_layout()
    Path("screenshots").mkdir(exist_ok=True)
    plt.savefig("screenshots/class_distribution.png", dpi=150)
    print("Graphique sauvegardé : screenshots/class_distribution.png")


if __name__ == "__main__":
    main()