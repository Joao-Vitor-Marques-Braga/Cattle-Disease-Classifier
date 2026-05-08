"""
clean_dataset.py — Consolida e limpa as classes do dataset de doenças bovinas.

Lê a estrutura de pastas do raw (dataset/organized ou dataset/raw) e cria um
dataset limpo em dataset_clean/ com apenas 3 classes bem definidas:
  - lumpy_skin  (agrupa múltiplas classes de lumpy skin)
  - ecthyma     (agrupa classes de ecthyma/BRD)
  - healthy     (saudável)
A classe "Unlabeled" é descartada.

Uso:
    python clean_dataset.py
"""

import random
import shutil
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv  # manter padrão do projeto

load_dotenv()

# ── Configuração — altere aqui se necessário ────────────────────────────────
# Pasta de origem (estrutura: <SRC_ROOT>/<split>/<classe>/<imagens>)
SRC_ROOT = Path("dataset") / "organized"

# Pasta de destino do dataset limpo
DEST_ROOT = Path("dataset_clean")

# Splits a processar
SPLITS = ["train", "val", "test"]

# Mapeamento: classe original → classe nova (None = descartar)
CLASS_MAP: dict[str, str | None] = {
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
    "Unlabeled":                        None,          # REMOVER
    "healthy lumpy skin":               "lumpy_skin",
}

# Extensões de imagem válidas
VALID_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
# ────────────────────────────────────────────────────────────────────────────


def safe_dest(dest_dir: Path, filename: str) -> Path:
    """
    Retorna um caminho de destino sem sobrescrita.
    Se <dest_dir>/<filename> já existir, acrescenta sufixo numérico:
      imagem.jpg → imagem_1.jpg → imagem_2.jpg → ...
    """
    dest = dest_dir / filename
    if not dest.exists():
        return dest

    stem = Path(filename).stem
    suffix = Path(filename).suffix
    counter = 1
    while True:
        new_name = f"{stem}_{counter}{suffix}"
        dest = dest_dir / new_name
        if not dest.exists():
            return dest
        counter += 1


def process_split(
    split: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Processa um split (train/val/test):
      - Copia imagens para as classes consolidadas em DEST_ROOT/<split>/
      - Descarta imagens da classe "Unlabeled" (ou qualquer classe com None no mapa)
    Retorna:
      counts_new   — {classe_nova: n_imagens_copiadas}
      discarded    — {classe_original: n_imagens_descartadas}
    """
    src_split = SRC_ROOT / split
    if not src_split.exists():
        print(f"  ⚠️  Split '{split}' não encontrado em {src_split} — pulando.")
        return {}, {}

    counts_new: dict[str, int] = defaultdict(int)
    discarded:  dict[str, int] = defaultdict(int)

    # Iterar sobre subpastas de classe
    for class_dir in sorted(src_split.iterdir()):
        if not class_dir.is_dir():
            continue

        orig_name = class_dir.name
        new_name  = CLASS_MAP.get(orig_name)  # None se não mapeado

        if new_name is None:
            # Classe descartada: contar imagens descartadas
            n = sum(1 for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VALID_EXT)
            if n:
                discarded[orig_name] += n
                print(f"    🗑  [{split}] '{orig_name}' → DESCARTADA ({n} imagens)")
            continue

        # Classe a copiar
        dest_dir = DEST_ROOT / split / new_name
        dest_dir.mkdir(parents=True, exist_ok=True)

        copied = 0
        for img in sorted(class_dir.iterdir()):
            if not (img.is_file() and img.suffix.lower() in VALID_EXT):
                continue
            dest_path = safe_dest(dest_dir, img.name)
            shutil.copy2(img, dest_path)
            copied += 1

        counts_new[new_name] += copied
        if copied:
            print(f"    ✅ [{split}] '{orig_name}' → '{new_name}' ({copied} imgs)")

    return dict(counts_new), dict(discarded)


def print_report(
    all_counts:   dict[str, dict[str, int]],
    all_discarded: dict[str, dict[str, int]],
) -> None:
    """Imprime tabela final com totais por classe e split."""
    new_classes = sorted({cls for sc in all_counts.values() for cls in sc})
    splits       = [s for s in SPLITS if s in all_counts]

    col_w   = 16
    sep_w   = col_w + 10 * len(splits) + 10

    print("\n" + "=" * sep_w)
    print("RELATÓRIO DO DATASET LIMPO")
    print("=" * sep_w)

    header = f"{'Classe':<{col_w}}" + "".join(f"{s:>10}" for s in splits) + f"{'Total':>10}"
    print(header)
    print("-" * sep_w)

    for cls in new_classes:
        row   = f"{cls:<{col_w}}"
        total = 0
        for s in splits:
            n = all_counts.get(s, {}).get(cls, 0)
            row  += f"{n:>10}"
            total += n
        row += f"{total:>10}"
        print(row)

    print("-" * sep_w)
    row   = f"{'TOTAL':<{col_w}}"
    grand = 0
    for s in splits:
        n = sum(all_counts.get(s, {}).values())
        row  += f"{n:>10}"
        grand += n
    row += f"{grand:>10}"
    print(row)
    print("=" * sep_w)

    # Imagens descartadas
    total_discarded = sum(
        n for sd in all_discarded.values() for n in sd.values()
    )
    if total_discarded:
        print(f"\n🗑  Imagens descartadas (Unlabeled/não mapeadas): {total_discarded}")
        for split, sd in all_discarded.items():
            for orig, n in sd.items():
                print(f"     [{split}] '{orig}': {n} imgs")
    else:
        print("\n🗑  Nenhuma imagem descartada.")

    print(f"\n✅ Dataset limpo salvo em: {DEST_ROOT.resolve()}")


def main() -> None:
    random.seed(42)

    # Limpar destino anterior para evitar mistura de versões
    if DEST_ROOT.exists():
        print(f"Removendo dataset_clean anterior em {DEST_ROOT}...")
        shutil.rmtree(DEST_ROOT)

    all_counts:    dict[str, dict[str, int]] = {}
    all_discarded: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        print(f"\nProcessando split: {split}")
        counts, discarded = process_split(split)
        if counts or discarded:
            all_counts[split]    = counts
            all_discarded[split] = discarded

    print_report(all_counts, all_discarded)


if __name__ == "__main__":
    main()
