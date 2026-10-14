import torch

print("Version PyTorch :", torch.__version__)
print("CUDA disponible :", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Carte graphique :", torch.cuda.get_device_name(0))

    x = torch.rand(3000, 3000, device="cuda")
    y = x @ x
    print("Calcul GPU réussi :", y.device)
else:
    print("PyTorch n'utilise pas encore le GPU.")