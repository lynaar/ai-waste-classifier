"""Entraînement de ResNet18 : head, finetune (layer4) ou finetune3 (layer3 + layer4)."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from data_utils import CLASSES, SEED, get_dataloaders
from model import (
    build_model,
    count_trainable,
    freeze_backbone,
    set_train_mode,
    unfreeze_layer3,
    unfreeze_layer4,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Une époque. Entraîne si un optimizer est fourni, sinon évalue."""
    training = optimizer is not None
    if training:
        set_train_mode(model)
    else:
        model.eval()

    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    acc = float(np.mean(np.array(all_preds) == np.array(all_labels)))
    macro_f1 = f1_score(all_labels, all_preds, average="macro")
    return total_loss / n, acc, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["head", "finetune", "finetune3"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=3,
                        help="époques avec la couche finale seule avant le dégel")
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--strong-aug", action="store_true",
                        help="augmentation de données plus forte (train seulement)")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--cosine", action="store_true",
                        help="learning rate décroissant (cosinus) après le dégel")
    parser.add_argument("--tag", default=None,
                        help="nom des fichiers de sortie (défaut : le mode)")
    args = parser.parse_args()

    name = args.tag or args.mode
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device} | Mode : {args.mode} | Tag : {name}")
    print(f"Augmentation forte : {args.strong_aug} | Label smoothing : {args.label_smoothing} "
          f"| Cosine : {args.cosine}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{name}.pth"
    history_path = out_dir / f"history_{name}.csv"

    train_loader, val_loader, _ = get_dataloaders(
        args.batch_size, args.num_workers, strong_aug=args.strong_aug
    )

    model = build_model().to(device)
    freeze_backbone(model)
    # Le label smoothing s'applique à l'entraînement ; la validation utilise la
    # cross-entropy standard, pour rester comparable aux configurations A et B.
    train_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    val_criterion = nn.CrossEntropyLoss()

    def make_optimizer(lr):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr)

    optimizer = make_optimizer(args.lr_head)
    scheduler = None
    print(f"Paramètres entraînables : {count_trainable(model):,}")

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        if args.mode != "head" and epoch == args.head_epochs + 1:
            if args.mode == "finetune3":
                unfreeze_layer3(model)
            unfreeze_layer4(model)
            optimizer = make_optimizer(args.lr_finetune)
            if args.cosine:
                scheduler = CosineAnnealingLR(
                    optimizer, T_max=args.epochs - args.head_epochs
                )
            print(f"--> dégel effectué, lr={args.lr_finetune}, "
                  f"paramètres entraînables : {count_trainable(model):,}")

        start = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, train_criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, val_criterion, device)
        if scheduler is not None:
            scheduler.step()
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
        })

        saved = ""
        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "mode": args.mode,
                "tag": name,
                "epoch": epoch,
                "val_macro_f1": va_f1,
                "val_acc": va_acc,
            }, ckpt_path)
            saved = " [meilleur, sauvegardé]"

        print(f"Epoch {epoch:2d}/{args.epochs} | train loss {tr_loss:.4f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.4f} acc {va_acc:.3f} F1 {va_f1:.3f} "
              f"| {time.time() - start:.0f}s{saved}")

    pd.DataFrame(history).to_csv(history_path, index=False)
    print(f"\nMeilleur F1 macro validation : {best_f1:.4f}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Historique : {history_path}")


if __name__ == "__main__":
    main()