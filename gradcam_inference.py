"""
gradcam_inference.py — Inferência com Grad-CAM para doenças bovinas.

Uso:
    python gradcam_inference.py <caminho_da_imagem>

Exemplo:
    python gradcam_inference.py imagens/vaca_teste.jpg

Saída:
    - Classe predita e probabilidades impressas no console
    - Mapa de calor Grad-CAM salvo como resultado_gradcam.jpg
"""

from dotenv import load_dotenv
load_dotenv()  # carrega .env antes de qualquer import dependente de env vars

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# ── Configurações ────────────────────────────────────────────────────────────
MODEL_PATH  = Path("best_model.pth")
OUTPUT_PATH = Path("resultado_gradcam.jpg")
IMAGE_SIZE  = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]
# ────────────────────────────────────────────────────────────────────────────


def load_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """
    Carrega o checkpoint salvo pelo train.py.
    Retorna o modelo em modo eval e a lista de nomes de classes.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em '{MODEL_PATH}'. "
            "Execute train.py primeiro para gerar o best_model.pth."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes  = checkpoint["num_classes"]

    # Recriar a arquitetura MobileNetV2 (mesma do train.py)
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(0.2),
        torch.nn.Linear(in_features, num_classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    """
    Carrega e pré-processa a imagem para inferência.
    Retorna o tensor normalizado e a imagem RGB normalizada para [0,1]
    (usada pela função de sobreposição do Grad-CAM).
    """
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    pil_img = Image.open(image_path).convert("RGB")
    tensor  = tf(pil_img).unsqueeze(0)  # shape: (1, 3, 224, 224)

    # Versão [0,1] sem normalização para sobreposição visual
    raw_tf  = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),          # valores em [0, 1]
    ])
    rgb_array = raw_tf(pil_img).permute(1, 2, 0).numpy()  # (H, W, 3)

    return tensor, rgb_array


def run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """
    Executa a inferência e retorna:
      - classe predita (str)
      - confiança da classe predita (float)
      - dicionário {classe: probabilidade} para todas as classes
    """
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze()  # shape: (num_classes,)

    idx_pred    = probs.argmax().item()
    class_pred  = class_names[idx_pred]
    confidence  = probs[idx_pred].item()
    all_probs   = {cls: probs[i].item() for i, cls in enumerate(class_names)}

    return class_pred, confidence, all_probs


def generate_gradcam(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    rgb_array: np.ndarray,
    target_class_idx: int,
    device: torch.device,
) -> np.ndarray:
    """
    Gera o mapa de calor Grad-CAM usando a última camada de features
    do MobileNetV2 (features[-1]) e o sobrepõe à imagem original.
    Retorna a imagem resultante como array numpy (H, W, 3) uint8.
    """
    # Camada alvo: features[-1] do MobileNetV2
    target_layers = [model.features[-1]]

    # A biblioteca pytorch-grad-cam gerencia hooks automaticamente
    cam = GradCAM(model=model, target_layers=target_layers)

    # Target para a classe predita
    targets = [ClassifierOutputTarget(target_class_idx)]

    # Gerar mapa de ativação (valores em [0, 1])
    grayscale_cam = cam(
        input_tensor=tensor.to(device),
        targets=targets,
    )[0]  # shape: (H, W)

    # rgb_array deve estar em [0, 1] — show_cam_on_image exige isso
    rgb_normalized = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)

    # Sobrepor o mapa de calor à imagem
    visualization = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)

    return visualization  # shape: (H, W, 3) uint8


def main() -> None:
    if len(sys.argv) < 2:
        print("Uso: python gradcam_inference.py <caminho_da_imagem>")
        print("Exemplo: python gradcam_inference.py imagem.jpg")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.exists():
        raise FileNotFoundError(f"Imagem não encontrada: '{image_path}'")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")

    # ── Carregar modelo ───────────────────────────────────────────────────────
    print(f"Carregando modelo de '{MODEL_PATH}'...")
    model, class_names = load_model(device)
    print(f"Classes do modelo: {class_names}")

    # ── Pré-processar imagem ──────────────────────────────────────────────────
    print(f"\nProcessando imagem: '{image_path}'")
    tensor, rgb_array = preprocess_image(image_path)

    # ── Inferência ────────────────────────────────────────────────────────────
    class_pred, confidence, all_probs = run_inference(model, tensor, class_names, device)
    idx_pred = class_names.index(class_pred)

    print("\n" + "=" * 50)
    print("RESULTADO DA INFERÊNCIA")
    print("=" * 50)
    print(f"  Classe predita : {class_pred}")
    print(f"  Confiança      : {confidence:.2%}")
    print("\n  Probabilidades por classe:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob * 30)
        print(f"    {cls:<25} {prob:.4f}  {bar}")
    print("=" * 50)

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    print("\nGerando mapa de calor Grad-CAM...")
    visualization = generate_gradcam(model, tensor, rgb_array, idx_pred, device)

    # Salvar resultado (OpenCV usa BGR)
    result_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(OUTPUT_PATH), result_bgr)
    print(f"✅ Resultado salvo em: '{OUTPUT_PATH.resolve()}'")


if __name__ == "__main__":
    main()
