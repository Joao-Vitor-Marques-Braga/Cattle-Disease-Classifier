"""
setup.py — Instala todas as dependências do pipeline de classificação de doenças bovinas.
Execute com: python setup.py
"""

import subprocess
import sys


def install(packages: list[str]) -> None:
    """Instala uma lista de pacotes via pip."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--upgrade", *packages]
    )


if __name__ == "__main__":
    dependencies = [
        # Framework de deep learning
        "torch",
        "torchvision",
        "torchaudio",

        # SDK do Roboflow para download de datasets
        "roboflow",

        # Grad-CAM para mapas de calor de interpretabilidade
        "grad-cam",

        # API REST
        "fastapi",
        "uvicorn[standard]",
        "python-multipart",  # necessário para upload de arquivos no FastAPI

        # Processamento de imagem
        "Pillow",
        "opencv-python",

        # Visualização
        "matplotlib",

        # Utilitários numéricos
        "numpy",
        "scikit-learn",  # para métricas precision/recall/F1
    ]

    print("=" * 60)
    print("Instalando dependências do pipeline de doenças bovinas...")
    print("=" * 60)

    install(dependencies)

    print("\n✅ Todas as dependências foram instaladas com sucesso!")
    print("\nPróximos passos:")
    print("  1. Defina a variável de ambiente ROBOFLOW_API_KEY")
    print("  2. Execute: python download_dataset.py")
    print("  3. Execute: python train.py")
    print("  4. Execute: python api.py")
