"""
download_dataset.py — Baixa o dataset "cattle-diseases" do Roboflow e organiza
em splits train / val / test compatíveis com torchvision.datasets.ImageFolder.

O Roboflow já entrega o dataset em train/valid/test com subpastas por classe.
Este script:
  1. Faz o download (se necessário)
  2. Copia para dataset/organized/ renomeando "valid" → "val"
  3. Imprime o resumo de imagens por classe e por split

Uso:
    python download_dataset.py

Variáveis de ambiente (pode vir do .env):
    ROBOFLOW_API_KEY  — sua chave de API do Roboflow
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from roboflow import Roboflow

# Carrega variáveis do arquivo .env (se existir) antes de qualquer leitura de env
load_dotenv()

# ── Configuração ─────────────────────────────────────────────────────────────
WORKSPACE    = "sliit-kuemd"
PROJECT      = "cattle-diseases"
VERSION      = 1           # altere se uma versão mais nova estiver disponível
EXPORT_FMT   = "folder"    # formato ImageFolder
RAW_DIR      = Path("dataset") / "raw"
ORGANIZED    = Path("dataset") / "organized"

# Mapeamento roboflow-split → nome padronizado
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}

# Extensões válidas (mesmas suportadas pelo torchvision.datasets.ImageFolder)
VALID_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────


def get_api_key() -> str:
    """Lê a chave de API do Roboflow da variável de ambiente (ou .env)."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise EnvironmentError(
            "A variável de ambiente ROBOFLOW_API_KEY não está definida.\n"
            "Defina-a no arquivo .env ou no terminal:\n"
            "  Windows PowerShell : $env:ROBOFLOW_API_KEY='sua_chave'\n"
            "  Linux / macOS      : export ROBOFLOW_API_KEY='sua_chave'"
        )
    return key


def download_raw(api_key: str) -> Path:
    """
    Faz o download do dataset via Roboflow SDK.
    Retorna o caminho da pasta raiz do download.
    """
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download(EXPORT_FMT, location=str(RAW_DIR))
    raw_path = Path(dataset.location)
    print(f"✅ Dataset baixado em: {raw_path}")
    return raw_path


def copy_organized(raw_path: Path) -> dict[str, dict[str, int]]:
    """
    Copia as imagens do raw para organized/train | val | test / <classe> / *.jpg
    O Roboflow já entrega com a estrutura: raw/<split>/<classe>/<imagens>
    Renomeia "valid" → "val" e ignora arquivos de texto.
    Retorna contagem de imagens {split: {classe: count}}.
    """
    counts: dict[str, dict[str, int]] = {}

    for rf_split, std_split in SPLIT_MAP.items():
        split_src = raw_path / rf_split
        if not split_src.exists():
            print(f"  ⚠️  Split '{rf_split}' não encontrado em {raw_path} — pulando.")
            continue

        counts[std_split] = defaultdict(int)

        # Cada subpasta de split_src é uma classe
        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            dest_dir = ORGANIZED / std_split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            # Copiar apenas arquivos de imagem
            for img in class_dir.iterdir():
                if img.is_file() and img.suffix.lower() in VALID_EXT:
                    shutil.copy2(img, dest_dir / img.name)
                    counts[std_split][class_name] += 1

    return counts


def print_summary(counts: dict[str, dict[str, int]]) -> None:
    """Imprime tabela resumindo imagens por classe e por split."""
    all_classes = sorted({cls for split_c in counts.values() for cls in split_c})
    splits = [s for s in ("train", "val", "test") if s in counts]

    header_parts = [f"{'Classe':<40}"] + [f"{s:>8}" for s in splits] + [f"{'Total':>8}"]
    sep_width = 40 + 9 * len(splits) + 9
    print("\n" + "=" * sep_width)
    print("RESUMO DO DATASET")
    print("=" * sep_width)
    print("".join(header_parts))
    print("-" * sep_width)

    for cls in all_classes:
        row = [f"{cls:<40}"]
        total = 0
        for s in splits:
            n = counts[s].get(cls, 0)
            row.append(f"{n:>8}")
            total += n
        row.append(f"{total:>8}")
        print("".join(row))

    print("-" * sep_width)
    row = [f"{'TOTAL':<40}"]
    grand = 0
    for s in splits:
        n = sum(counts[s].values())
        row.append(f"{n:>8}")
        grand += n
    row.append(f"{grand:>8}")
    print("".join(row))
    print("=" * sep_width)


def main() -> None:
    api_key = get_api_key()

    print(f"Baixando dataset '{PROJECT}' do workspace '{WORKSPACE}'...")
    raw_path = download_raw(api_key)

    # Limpar organized anterior para evitar mistura de versões
    if ORGANIZED.exists():
        print(f"\nRemovendo organized anterior em {ORGANIZED}...")
        shutil.rmtree(ORGANIZED)

    print(f"\nCopiando e organizando imagens para {ORGANIZED}...")
    counts = copy_organized(raw_path)

    if not counts:
        raise RuntimeError(
            "Nenhuma imagem foi copiada. Verifique a estrutura do dataset baixado."
        )

    print_summary(counts)
    print(f"\n✅ Dataset organizado em: {ORGANIZED.resolve()}")
    print("   Estrutura: organized/train | val | test / <classe> / *.jpg")


if __name__ == "__main__":
    main()
