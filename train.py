"""
train.py — Fine-tuning do MobileNetV2 para classificação de doenças bovinas.

Uso:
    python train.py

O script lê o dataset da pasta dataset/organized/ (gerada pelo download_dataset.py),
treina o modelo e salva o melhor checkpoint como best_model.pth.
"""

from dotenv import load_dotenv
load_dotenv()  # carrega .env antes de qualquer import dependente de env vars

import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report

# ── Reprodutibilidade ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Configurações de treino ──────────────────────────────────────────────────
DATASET_DIR   = Path("dataset") / "organized"
MODEL_PATH    = Path("best_model.pth")
CURVES_PATH   = Path("training_curves.png")
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
PATIENCE_LR   = 3    # épocas sem melhora antes de reduzir lr
PATIENCE_STOP = 5    # épocas sem melhora para early stopping

# Normalização ImageNet (esperada pelo MobileNetV2 pré-treinado)
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
# ────────────────────────────────────────────────────────────────────────────


def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """Retorna transformações de treino (com augmentation) e validação/teste."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, val_tf


def load_datasets(train_tf, val_tf) -> tuple[
    datasets.ImageFolder, datasets.ImageFolder, datasets.ImageFolder
]:
    """Carrega os três splits como ImageFolder."""
    train_ds = datasets.ImageFolder(str(DATASET_DIR / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(DATASET_DIR / "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(str(DATASET_DIR / "test"),  transform=val_tf)
    return train_ds, val_ds, test_ds


def build_model(num_classes: int) -> nn.Module:
    """
    Carrega MobileNetV2 pré-treinado e realiza fine-tuning parcial:
      • Congela todas as camadas.
      • Descongela as últimas 2 camadas de features + classifier head.
      • Substitui o classifier head pelo número de classes do dataset.
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

    # Congelar todos os parâmetros
    for param in model.parameters():
        param.requires_grad = False

    # Descongelar as 2 últimas camadas do bloco features
    features_list = list(model.features.children())
    for layer in features_list[-2:]:
        for param in layer.parameters():
            param.requires_grad = True

    # Substituir o classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    # O novo head é treinável por padrão

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Executa uma época de treino. Retorna (loss_média, accuracy)."""
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Avalia o modelo sem gradiente. Retorna (loss_média, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def test_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_names: list[str],
) -> None:
    """Imprime relatório completo de classificação (precision/recall/F1 por classe)."""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    print("\n" + "=" * 60)
    print("RELATÓRIO DE CLASSIFICAÇÃO — CONJUNTO DE TESTE")
    print("=" * 60)
    print(classification_report(all_labels, all_preds, target_names=class_names))


def save_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
) -> None:
    """Plota e salva os gráficos de loss e accuracy por época."""
    epochs_range = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Gráfico de Loss
    axes[0].plot(epochs_range, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs_range, val_losses, label="Val Loss", marker="o")
    axes[0].set_title("Loss por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Gráfico de Accuracy
    axes[1].plot(epochs_range, train_accs, label="Train Acc", marker="o")
    axes[1].plot(epochs_range, val_accs, label="Val Acc", marker="o")
    axes[1].set_title("Accuracy por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(CURVES_PATH)
    plt.close(fig)
    print(f"\n📊 Curvas de treinamento salvas em: {CURVES_PATH.resolve()}")


def main() -> None:
    # ── Dispositivo ──────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # ── Dados ────────────────────────────────────────────────────────────────
    train_tf, val_tf = build_transforms()
    train_ds, val_ds, test_ds = load_datasets(train_tf, val_tf)

    # Classes são lidas dinamicamente do dataset (não hardcoded)
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"\nClasses detectadas ({num_classes}): {class_names}")

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=(device.type == "cuda")
    )

    # ── Modelo, otimizador, loss, scheduler ──────────────────────────────────
    model = build_model(num_classes).to(device)
    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_LR)

    # ── Histórico de métricas ────────────────────────────────────────────────
    train_losses, val_losses = [], []
    train_accs, val_accs     = [], []

    best_val_loss  = float("inf")
    epochs_no_improve = 0
    best_epoch     = 0

    print("\nIniciando treinamento...\n")
    print(f"{'Época':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Tempo':>7}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)

        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(
            f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.4f} "
            f"{val_loss:>10.4f} {val_acc:>9.4f} {current_lr:>10.1e} {elapsed:>6.1f}s"
        )

        # Salvar melhor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
            # Salvar junto com metadados necessários para inferência
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "class_names": class_names,
                    "num_classes": num_classes,
                },
                MODEL_PATH,
            )
            print(f"  ✅ Novo melhor modelo salvo (val_loss={val_loss:.4f})")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= PATIENCE_STOP:
            print(f"\n⏹  Early stopping na época {epoch} (sem melhora por {PATIENCE_STOP} épocas).")
            print(f"   Melhor modelo: época {best_epoch}, val_loss={best_val_loss:.4f}")
            break

    # ── Curvas de treinamento ─────────────────────────────────────────────────
    save_curves(train_losses, val_losses, train_accs, val_accs)

    # ── Avaliação no conjunto de teste ────────────────────────────────────────
    print(f"\nCarregando melhor modelo (época {best_epoch}) para avaliação no teste...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_model(model, test_loader, device, class_names)
    print(f"\n✅ Treinamento concluído. Modelo salvo em: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
