"""Étape 2 : crée le split train/val/test, le vérifie et le sauvegarde."""

import pandas as pd

from data_utils import (
    CLASSES,
    check_no_leakage,
    collect_images,
    get_dataloaders,
    make_splits,
    save_splits,
)


def main() -> None:
    df = collect_images()
    train_df, val_df, test_df = make_splits(df)
    check_no_leakage(train_df, val_df, test_df)
    save_splits(train_df, val_df, test_df)

    print(f"Total : {len(df)} images")
    print(f"Train : {len(train_df)} | Val : {len(val_df)} | Test : {len(test_df)}\n")

    table = pd.DataFrame(
        {
            "train": train_df["label"].value_counts(),
            "val": val_df["label"].value_counts(),
            "test": test_df["label"].value_counts(),
        }
    ).reindex(CLASSES)
    print("Images par classe et par ensemble :")
    print(table, "\n")
    print("Aucune fuite entre les ensembles : OK")

    # Test du pipeline : on charge un batch de chaque DataLoader
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)
    images, labels = next(iter(train_loader))
    print(f"\nBatch train : images {tuple(images.shape)}, labels {tuple(labels.shape)}")
    print(f"Valeurs des pixels : min={images.min():.2f}, max={images.max():.2f}")
    print("Splits sauvegardés dans data/splits/")


if __name__ == "__main__":
    main()