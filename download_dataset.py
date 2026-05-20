"""
download_dataset.py — Baixa os datasets do Roboflow e organiza
em splits train / val / test compatíveis com torchvision.datasets.ImageFolder.

O script iterará sobre a configuração de datasets (Estágios e Pragas),
fazendo o download de cada um e organizando em pastas separadas.
"""

import os
import shutil
from pathlib import Path
from collections import defaultdict

from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

# ── Configuração ─────────────────────────────────────────────────────────────
DATASETS = [
    {
        "name": "stages",
        "workspace": "capstone-1ngby",
        "project": "capstone-maize-growth",
        "version": 2,
        "format": "clip"
    },
    {
        "name": "pests",
        "workspace": "cornpest",
        "project": "corn-pest-v4-ybizr",
        "version": 7,
        "format": "clip"
    }
]

BASE_DIR = Path("dataset")
RAW_DIR = BASE_DIR / "raw"

SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test":  "test",
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}
# ─────────────────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise EnvironmentError("ROBOFLOW_API_KEY não definida.")
    return key


def download_raw(api_key: str, ds_config: dict) -> Path:
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(ds_config["workspace"]).project(ds_config["project"])
    ds_raw_dir = RAW_DIR / ds_config["name"]
    dataset = project.version(ds_config["version"]).download(
        ds_config["format"], location=str(ds_raw_dir)
    )
    raw_path = Path(dataset.location)
    print(f"✅ Dataset '{ds_config['name']}' baixado em: {raw_path}")
    return raw_path


def copy_organized(raw_path: Path, organized_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}

    for rf_split, std_split in SPLIT_MAP.items():
        split_src = raw_path / rf_split
        if not split_src.exists():
            continue

        counts[std_split] = defaultdict(int)

        for class_dir in sorted(split_src.iterdir()):
            if not class_dir.is_dir():
                continue

            class_name = class_dir.name
            dest_dir = organized_dir / std_split / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)

            for img in class_dir.iterdir():
                if img.is_file() and img.suffix.lower() in VALID_EXT:
                    shutil.copy2(img, dest_dir / img.name)
                    counts[std_split][class_name] += 1

    return counts


def print_summary(name: str, counts: dict[str, dict[str, int]]) -> None:
    all_classes = sorted({cls for split_c in counts.values() for cls in split_c})
    splits = [s for s in ("train", "val", "test") if s in counts]

    header_parts = [f"{'Classe':<40}"] + [f"{s:>8}" for s in splits] + [f"{'Total':>8}"]
    sep_width = 40 + 9 * len(splits) + 9
    print("\n" + "=" * sep_width)
    print(f"RESUMO DO DATASET: {name.upper()}")
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

    for ds_config in DATASETS:
        name = ds_config["name"]
        print(f"\n🚀 Processando dataset: {name}")
        
        raw_path = download_raw(api_key, ds_config)
        organized_dir = BASE_DIR / f"organized_{name}"

        if organized_dir.exists():
            print(f"Removendo pastas antigas de {organized_dir}...")
            shutil.rmtree(organized_dir)

        print(f"Organizando imagens para {organized_dir}...")
        counts = copy_organized(raw_path, organized_dir)

        if not counts:
            print(f"⚠️  Nenhuma imagem foi copiada para o dataset {name}.")
        else:
            print_summary(name, counts)
            print(f"✅ Dataset {name} finalizado: {organized_dir.resolve()}")


if __name__ == "__main__":
    main()
