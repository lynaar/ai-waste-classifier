"""Étape 2 : préparation des données (split, transformations, DataLoaders)."""

from pathlib import Path

import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DATA_DIR = Path("data/dataset-resized")
SPLITS_DIR = Path("data/splits")
CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

SEED = 42
IMAGE_SIZE = 224
# Moyenne et écart-type d'ImageNet : obligatoires pour ResNet18 pré-entraîné
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def collect_images(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Retourne un DataFrame (path, label, label_idx) de toutes les images."""
    rows = []
    for class_name in CLASSES:
        class_dir = data_dir / class_name
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append(
                    {
                        "path": path.as_posix(),
                        "label": class_name,
                        "label_idx": CLASS_TO_IDX[class_name],
                    }
                )
    return pd.DataFrame(rows)


def make_splits(df: pd.DataFrame, seed: int = SEED):
    """Split stratifié 70% / 15% / 15%."""
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label_idx"], random_state=seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label_idx"], random_state=seed
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def check_no_leakage(train_df, val_df, test_df) -> None:
    """Vérifie qu'aucune image n'est présente dans deux ensembles."""
    train, val, test = set(train_df["path"]), set(val_df["path"]), set(test_df["path"])
    assert train.isdisjoint(val), "Fuite entre train et val"
    assert train.isdisjoint(test), "Fuite entre train et test"
    assert val.isdisjoint(test), "Fuite entre val et test"


def save_splits(train_df, val_df, test_df, splits_dir: Path = SPLITS_DIR) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)


def load_splits(splits_dir: Path = SPLITS_DIR):
    return (
        pd.read_csv(splits_dir / "train.csv"),
        pd.read_csv(splits_dir / "val.csv"),
        pd.read_csv(splits_dir / "test.csv"),
    )


def get_transforms(train: bool, strong: bool = False):
    """Augmentation seulement pour train ; resize + normalisation pour val/test."""
    if train and strong:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.4, 1.0), ratio=(0.75, 1.33)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(25),
                transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.4, hue=0.05),
                transforms.RandomGrayscale(p=0.1),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        )
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class WasteDataset(Dataset):
    """Dataset PyTorch construit à partir d'un DataFrame (path, label_idx)."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label_idx"])


def get_dataloaders(batch_size: int = 32, num_workers: int = 0, strong_aug: bool = False):
    """Charge les CSV et retourne les DataLoaders train / val / test."""
    train_df, val_df, test_df = load_splits()
    train_ds = WasteDataset(train_df, get_transforms(train=True, strong=strong_aug))
    val_ds = WasteDataset(val_df, get_transforms(train=False))
    test_ds = WasteDataset(test_df, get_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, test_loader