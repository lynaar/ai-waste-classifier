"""Étape 3 : entraînement de ResNet18 (config A : head, config B : finetune)."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn

from data_utils import CLASSES, SEED, get_dataloaders
from model import (
    build_model,
    count_trainable,
    freeze_backbone,
    set_train_mode,
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
    parser.add_argument("--mode", choices=["head", "finetune"], required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--head-epochs", type=int, default=3,
                        help="finetune : époques avec la couche finale seule")
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device} | Mode : {args.mode}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"best_{args.mode}.pth"
    history_path = out_dir / f"history_{args.mode}.csv"

    train_loader, val_loader, _ = get_dataloaders(args.batch_size, args.num_workers)

    model = build_model().to(device)
    freeze_backbone(model)
    criterion = nn.CrossEntropyLoss()

    def make_optimizer(lr):
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.Adam(params, lr=lr)

    optimizer = make_optimizer(args.lr_head)
    print(f"Paramètres entraînables : {count_trainable(model):,}")

    best_f1, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        # Config B : après quelques époques, on dégèle layer4
        if args.mode == "finetune" and epoch == args.head_epochs + 1:
            unfreeze_layer4(model)
            optimizer = make_optimizer(args.lr_finetune)
            print(f"--> layer4 dégelé, lr={args.lr_finetune}, "
                  f"paramètres entraînables : {count_trainable(model):,}")

        start = time.time()
        tr_loss, tr_acc, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)
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