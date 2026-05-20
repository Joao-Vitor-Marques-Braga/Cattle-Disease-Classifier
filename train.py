"""
train.py — Fine-tuning do MobileNetV2 para classificação de estágios e pragas do milho.

Uso:
    python train.py

O script lê os datasets gerados pelo download_dataset.py,
treina os modelos sequencialmente e salva os checkpoints.
"""

from dotenv import load_dotenv
load_dotenv()

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

# ── Configurações de treino globais ──────────────────────────────────────────
IMAGE_SIZE    = 224
BATCH_SIZE    = 32
EPOCHS        = 20
LR            = 1e-4
PATIENCE_LR   = 3    # épocas sem melhora antes de reduzir lr
PATIENCE_STOP = 5    # épocas sem melhora para early stopping

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
# ────────────────────────────────────────────────────────────────────────────

def build_transforms() -> tuple[transforms.Compose, transforms.Compose]:
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


def load_datasets(dataset_dir: Path, train_tf, val_tf) -> tuple:
    train_ds = datasets.ImageFolder(str(dataset_dir / "train"), transform=train_tf)
    val_ds   = datasets.ImageFolder(str(dataset_dir / "val"),   transform=val_tf)
    test_ds  = datasets.ImageFolder(str(dataset_dir / "test"),  transform=val_tf)
    return train_ds, val_ds, test_ds


def build_model(num_classes: int) -> nn.Module:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    features_list = list(model.features.children())
    for layer in features_list[-2:]:
        for param in layer.parameters():
            param.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(model, loader, optimizer, criterion, device):
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
def evaluate(model, loader, criterion, device):
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
def test_model(model, loader, device, class_names):
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


def save_curves(train_losses, val_losses, train_accs, val_accs, curves_path: Path):
    epochs_range = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(epochs_range, train_losses, label="Train Loss", marker="o")
    axes[0].plot(epochs_range, val_losses, label="Val Loss", marker="o")
    axes[0].set_title("Loss por Época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(epochs_range, train_accs, label="Train Acc", marker="o")
    axes[1].plot(epochs_range, val_accs, label="Val Acc", marker="o")
    axes[1].set_title("Accuracy por Época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(curves_path)
    plt.close(fig)


def run_training(name: str, device: torch.device):
    dataset_dir = Path("dataset") / f"organized_{name}"
    model_path = Path(f"best_model_{name}.pth")
    curves_path = Path(f"training_curves_{name}.png")

    if not dataset_dir.exists():
        print(f"\n⚠️ Dataset não encontrado para '{name}' em {dataset_dir}. Pule o treinamento.")
        return

    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO TREINAMENTO: {name.upper()}")
    print(f"{'='*60}")

    train_tf, val_tf = build_transforms()
    train_ds, val_ds, test_ds = load_datasets(dataset_dir, train_tf, val_tf)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"\nClasses detectadas ({num_classes}): {class_names}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    model = build_model(num_classes).to(device)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=PATIENCE_LR)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_epoch = 0

    print(f"\n{'Época':>6} {'Train Loss':>11} {'Train Acc':>10} {'Val Loss':>10} {'Val Acc':>9} {'LR':>10} {'Tempo':>7}")
    print("-" * 70)

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"{epoch:>6} {train_loss:>11.4f} {train_acc:>9.4f} {val_loss:>10.4f} {val_acc:>9.4f} {current_lr:>10.1e} {elapsed:>6.1f}s")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_no_improve = 0
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
                model_path,
            )
            print(f"  ✅ Novo melhor modelo salvo ({model_path.name})")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE_STOP:
            print(f"\n⏹ Early stopping na época {epoch}. Melhor: época {best_epoch}.")
            break

    save_curves(train_losses, val_losses, train_accs, val_accs, curves_path)

    print("\nCarregando melhor modelo para teste...")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_model(model, test_loader, device, class_names)
    print(f"\n✅ Treinamento de {name.upper()} concluído!")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")
    
    # Treina ambos os modelos
    run_training("stages", device)
    run_training("pests", device)


if __name__ == "__main__":
    main()
