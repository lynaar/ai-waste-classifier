"""Étape 3 : modèle ResNet18 (transfert d'apprentissage)."""

import torch.nn as nn
from torchvision import models

NUM_CLASSES = 6


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """ResNet18 avec poids ImageNet et dernière couche remplacée."""
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 512 -> 6
    return model


def freeze_backbone(model: nn.Module) -> None:
    """Gèle tout sauf la couche finale (fc)."""
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_layer4(model: nn.Module) -> None:
    """Dégèle le dernier bloc convolutif (layer4) pour le fine-tuning."""
    for param in model.layer4.parameters():
        param.requires_grad = True

def unfreeze_layer3(model: nn.Module) -> None:
    """Dégèle aussi le bloc layer3 (fine-tuning plus profond)."""
    for param in model.layer3.parameters():
        param.requires_grad = True

        
def set_train_mode(model: nn.Module) -> None:
    """Mode entraînement, mais les couches BatchNorm gelées restent en mode eval.

    Sans cela, les statistiques BatchNorm des couches gelées seraient
    quand même modifiées pendant l'entraînement.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            params = list(module.parameters())
            if params and not params[0].requires_grad:
                module.eval()


def count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Petit test : python src/model.py
    model = build_model()
    total = sum(p.numel() for p in model.parameters())
    print(f"Paramètres totaux : {total:,}")
    freeze_backbone(model)
    print(f"Entraînables (fc seule) : {count_trainable(model):,}")
    unfreeze_layer4(model)
    print(f"Entraînables (fc + layer4) : {count_trainable(model):,}")