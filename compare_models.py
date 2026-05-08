"""
compare_models.py — Compara o modelo original (7 classes) com o modelo limpo (3 classes).

Avalia ambos os modelos no conjunto de teste do dataset_clean/ e imprime uma
tabela comparativa com accuracy, F1 por classe e conclusão automática.

Uso:
    python compare_models.py
"""

import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, f1_score

load_dotenv()

# ── Reprodutibilidade ─────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Configuração — altere aqui se necessário ──────────────────────────────────
ORIGINAL_MODEL_PATH = Path("best_model.pth")        # modelo original (7 classes)
CLEAN_MODEL_PATH    = Path("best_model_clean.pth")  # modelo limpo (3 classes)
TEST_DIR            = Path("dataset_clean") / "test" # conjunto de teste compartilhado

IMAGE_SIZE = 224
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
# ─────────────────────────────────────────────────────────────────────────────

# Mapeamento para colapsar predições do modelo original (7 classes) nas 3 classes limpas.
# Se uma classe original não tiver equivalente no dataset limpo, é ignorada.
ORIG_TO_CLEAN: dict[str, str] = {
    "lumpy skin":                       "lumpy_skin",
    "Contagious Dermatitis lumpy skin": "lumpy_skin",
    "Dermatitis Ecthym lumpy skin":     "lumpy_skin",
    "Ecthym skin":                      "ecthyma",
    "(BRD) Disease Ecthym":             "ecthyma",
    "(BRD) Bovine Dermatitis Disease healthy lumpy": "lumpy_skin",
    "(BRD) Bovine Disease Respiratory": "ecthyma",
    "Contagious Ecthym":                "ecthyma",
    "Dermatitis":                       "ecthyma",
    "healthy":                          "healthy",
    "healthy lumpy skin":               "lumpy_skin",
    # "Unlabeled" → não tem mapeamento, será ignorado
}


def get_val_transform() -> transforms.Compose:
    """Transformação padrão de inferência (sem augmentation)."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """
    Carrega um checkpoint salvo pelo train.py / retrain.py.
    Retorna o modelo em modo eval e a lista de nomes de classes.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: '{path}'. "
            "Execute train.py (ou retrain.py) primeiro."
        )
    ckpt        = torch.load(path, map_location=device)
    class_names = ckpt["class_names"]
    num_classes  = ckpt["num_classes"]

    model = models.mobilenet_v2(weights=None)
    in_feat = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_feat, num_classes),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, class_names


@torch.no_grad()
def predict_all(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Roda inferência em todo o DataLoader. Retorna (predições, rótulos reais)."""
    all_preds, all_labels = [], []
    for images, labels in loader:
        preds = model(images.to(device)).argmax(1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())
    return all_preds, all_labels


def remap_original_preds(
    preds_idx: list[int],
    orig_classes: list[str],
    clean_classes: list[str],
) -> list[int | None]:
    """
    Converte índices de predição do modelo original (7 classes) para índices
    do espaço de classes limpas (3 classes).
    Retorna None para predições sem mapeamento (ex: Unlabeled).
    """
    clean_idx_map = {cls: i for i, cls in enumerate(clean_classes)}
    remapped = []
    for idx in preds_idx:
        orig_name  = orig_classes[idx]
        clean_name = ORIG_TO_CLEAN.get(orig_name)
        remapped.append(clean_idx_map.get(clean_name))  # None se sem mapeamento
    return remapped


def print_comparison(
    clean_classes: list[str],
    labels_true:   list[int],
    preds_clean:   list[int],
    preds_orig_remapped: list[int | None],
) -> None:
    """
    Imprime a tabela comparativa lado a lado:
      - Accuracy geral
      - F1 por classe
      - Conclusão automática
    """
    # Filtrar amostras sem mapeamento (None) do modelo original
    valid_mask = [p is not None for p in preds_orig_remapped]
    labels_filt = [labels_true[i]          for i, v in enumerate(valid_mask) if v]
    orig_filt   = [preds_orig_remapped[i]  for i, v in enumerate(valid_mask) if v]
    clean_filt  = [preds_clean[i]          for i, v in enumerate(valid_mask) if v]

    acc_orig  = accuracy_score(labels_filt, orig_filt)
    acc_clean = accuracy_score(labels_filt, clean_filt)

    f1_orig  = f1_score(labels_filt, orig_filt,  average=None,
                        labels=list(range(len(clean_classes))), zero_division=0)
    f1_clean = f1_score(labels_filt, clean_filt, average=None,
                        labels=list(range(len(clean_classes))), zero_division=0)

    f1_macro_orig  = f1_score(labels_filt, orig_filt,  average="macro", zero_division=0)
    f1_macro_clean = f1_score(labels_filt, clean_filt, average="macro", zero_division=0)

    # ── Tabela comparativa ───────────────────────────────────────────────────
    col = 22
    sep = "=" * (col + 28)
    print("\n" + sep)
    print("COMPARAÇÃO DE MODELOS")
    print(sep)
    print(f"{'Métrica':<{col}} {'Original (7 cls)':>13} {'Limpo (3 cls)':>13}")
    print("-" * (col + 28))
    print(f"{'Accuracy geral':<{col}} {acc_orig:>13.4f} {acc_clean:>13.4f}")
    print(f"{'F1 Macro':<{col}} {f1_macro_orig:>13.4f} {f1_macro_clean:>13.4f}")
    print("-" * (col + 28))

    for i, cls in enumerate(clean_classes):
        fo = f1_orig[i]  if i < len(f1_orig)  else 0.0
        fc = f1_clean[i] if i < len(f1_clean) else 0.0
        print(f"  F1 [{cls:<{col-6}}] {fo:>13.4f} {fc:>13.4f}")

    print("=" * (col + 28))

    # ── Relatório detalhado ───────────────────────────────────────────────────
    print("\n── Relatório detalhado — Modelo Original (remapeado) ──")
    print(classification_report(labels_filt, orig_filt,
                                target_names=clean_classes, zero_division=0))

    print("── Relatório detalhado — Modelo Limpo ──")
    print(classification_report(labels_filt, clean_filt,
                                target_names=clean_classes, zero_division=0))

    # ── Conclusão automática ──────────────────────────────────────────────────
    print("=" * (col + 28))
    print("CONCLUSÃO AUTOMÁTICA")
    print("=" * (col + 28))

    if acc_clean > acc_orig and f1_macro_clean > f1_macro_orig:
        winner = "MODELO LIMPO (3 classes)"
        reason = (
            f"obteve accuracy {acc_clean:.2%} vs {acc_orig:.2%} (+{(acc_clean-acc_orig)*100:.1f}pp) "
            f"e F1 macro {f1_macro_clean:.4f} vs {f1_macro_orig:.4f}. "
            "A consolidação de classes redundantes melhorou a discriminação do modelo."
        )
    elif acc_orig > acc_clean and f1_macro_orig > f1_macro_clean:
        winner = "MODELO ORIGINAL (7 classes)"
        reason = (
            f"obteve accuracy {acc_orig:.2%} vs {acc_clean:.2%} "
            f"e F1 macro {f1_macro_orig:.4f} vs {f1_macro_clean:.4f}. "
            "O modelo original, mesmo com classes redundantes, generaliza melhor neste conjunto."
        )
    elif f1_macro_clean >= f1_macro_orig:
        winner = "MODELO LIMPO (3 classes)"
        reason = (
            f"apresenta F1 macro superior ou igual ({f1_macro_clean:.4f} vs {f1_macro_orig:.4f}) "
            "com muito menos classes, tornando a API mais simples e interpretável."
        )
    else:
        winner = "EMPATE — resultados muito próximos"
        reason = (
            "Ambos os modelos têm desempenho similar. "
            "Recomendamos o modelo limpo pela maior clareza das classes."
        )

    print(f"  🏆 Vencedor: {winner}")
    print(f"  ℹ️  Motivo : {reason}")
    print("=" * (col + 28))

    if len(valid_mask) - sum(valid_mask):
        ignored = len(valid_mask) - sum(valid_mask)
        print(f"\n  ⚠️  {ignored} amostra(s) ignorada(s) na comparação do modelo original "
              f"(preditas como classe sem mapeamento, ex: Unlabeled).")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}\n")

    # ── Carregar modelos ──────────────────────────────────────────────────────
    print(f"Carregando modelo original: {ORIGINAL_MODEL_PATH}")
    model_orig, orig_classes = load_checkpoint(ORIGINAL_MODEL_PATH, device)

    print(f"Carregando modelo limpo:    {CLEAN_MODEL_PATH}")
    model_clean, clean_classes = load_checkpoint(CLEAN_MODEL_PATH, device)

    print(f"\nClasses originais ({len(orig_classes)}): {orig_classes}")
    print(f"Classes limpas   ({len(clean_classes)}): {clean_classes}")

    # ── Dataset de teste (dataset_clean/test) ─────────────────────────────────
    val_tf   = get_val_transform()
    test_ds  = datasets.ImageFolder(str(TEST_DIR), transform=val_tf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"\nConjunto de teste: {len(test_ds)} imagens em {len(test_ds.classes)} classes")

    # ── Inferência ────────────────────────────────────────────────────────────
    print("\nRodando inferência — modelo limpo...")
    preds_clean, labels_true = predict_all(model_clean, test_loader, device)

    print("Rodando inferência — modelo original...")
    preds_orig_raw, _ = predict_all(model_orig, test_loader, device)

    # Converter predições do modelo original para o espaço de 3 classes
    preds_orig_remapped = remap_original_preds(preds_orig_raw, orig_classes, clean_classes)

    # ── Comparação ────────────────────────────────────────────────────────────
    print_comparison(clean_classes, labels_true, preds_clean, preds_orig_remapped)


if __name__ == "__main__":
    main()
