"""
api.py — Servidor FastAPI para inferência de doenças bovinas.

Uso:
    python api.py
    # ou com uvicorn diretamente:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoint principal:
    POST /predict  — recebe uma imagem e retorna classe, confiança e heatmap Grad-CAM.
"""

from dotenv import load_dotenv
load_dotenv()  # carrega .env antes de qualquer import dependente de env vars

import base64
import io
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import models, transforms

# ── Configurações ────────────────────────────────────────────────────────────
MODEL_PATH = Path("best_model.pth")
IMAGE_SIZE  = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# ────────────────────────────────────────────────────────────────────────────

# ── Estado global do modelo (carregado uma única vez na inicialização) ────────
_model_state: dict[str, Any] = {
    "model": None,
    "class_names": None,
    "device": None,
}


# ── Schemas de resposta ───────────────────────────────────────────────────────
class PredictResponse(BaseModel):
    doenca_detectada: str
    confianca: float
    probabilidades: dict[str, float]
    heatmap_base64: str


# ── Ciclo de vida: carrega modelo ao iniciar o servidor ──────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega (e descarrega) o modelo durante o ciclo de vida da aplicação."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de inferência: {device}")

    if not MODEL_PATH.exists():
        logger.warning(
            f"AVISO: Modelo não encontrado em '{MODEL_PATH}'. "
            "O endpoint /predict retornará erro 503 até que o modelo seja treinado."
        )
    else:
        try:
            checkpoint   = torch.load(MODEL_PATH, map_location=device)
            class_names  = checkpoint["class_names"]
            num_classes  = checkpoint["num_classes"]

            # Recriar arquitetura idêntica à do train.py
            model = models.mobilenet_v2(weights=None)
            in_feat = model.classifier[1].in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(0.2),
                torch.nn.Linear(in_feat, num_classes),
            )
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()

            _model_state["model"]       = model
            _model_state["class_names"] = class_names
            _model_state["device"]      = device

            logger.info(f"Modelo carregado — {num_classes} classes: {class_names}")
        except Exception as exc:
            logger.error(f"Falha ao carregar o modelo: {exc}")

    yield  # A aplicação roda aqui

    # Limpeza ao encerrar
    _model_state["model"] = None
    logger.info("Aplicação encerrada — modelo descarregado.")


# ── Criação da aplicação ─────────────────────────────────────────────────────
app = FastAPI(
    title="API de Detecção de Doenças Bovinas",
    description="Classifica doenças bovinas e retorna mapa de calor Grad-CAM.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS liberado para qualquer origem (integração com app Android)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # em produção, substitua por origens específicas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utilitários de processamento de imagem ───────────────────────────────────

def _bytes_to_pil(data: bytes) -> Image.Image:
    """Converte bytes de upload para imagem PIL RGB."""
    return Image.open(io.BytesIO(data)).convert("RGB")


def _preprocess(pil_img: Image.Image) -> tuple[torch.Tensor, np.ndarray]:
    """
    Retorna:
      tensor     — tensor normalizado (1, 3, H, W) para inferência
      rgb_array  — imagem [0,1] float32 para sobreposição Grad-CAM
    """
    tf_norm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    tf_raw = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    tensor    = tf_norm(pil_img).unsqueeze(0)
    rgb_array = tf_raw(pil_img).permute(1, 2, 0).numpy()
    return tensor, rgb_array


def _run_inference(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
) -> tuple[str, float, dict[str, float]]:
    """Executa forward pass e retorna classe, confiança e probabilidades."""
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = F.softmax(logits, dim=1).squeeze()

    idx        = probs.argmax().item()
    all_probs  = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    return class_names[idx], float(probs[idx]), all_probs


def _generate_heatmap_b64(
    model: torch.nn.Module,
    tensor: torch.Tensor,
    rgb_array: np.ndarray,
    target_idx: int,
    device: torch.device,
) -> str:
    """
    Gera o mapa Grad-CAM, sobrepõe à imagem original e retorna em base64 JPEG.
    """
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_idx)]

    grayscale_cam   = cam(input_tensor=tensor.to(device), targets=targets)[0]
    rgb_normalized  = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)
    visualization   = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)

    # Codificar para base64 via buffer em memória (sem gravar em disco)
    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


# ── Endpoint principal ────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, summary="Classifica doença bovina")
async def predict(file: UploadFile = File(..., description="Imagem do animal (JPG/PNG)")):
    """
    Recebe uma imagem via multipart/form-data e retorna:
    - doenca_detectada: nome da classe com maior probabilidade
    - confianca: probabilidade da classe predita
    - probabilidades: dicionário {classe: probabilidade} para todas as classes
    - heatmap_base64: mapa de calor Grad-CAM em base64 (JPEG)
    """
    # Verificar se o modelo está disponível
    if _model_state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Modelo não carregado. Verifique se o best_model.pth existe "
                "e se o servidor foi iniciado corretamente."
            ),
        )

    model       = _model_state["model"]
    class_names = _model_state["class_names"]
    device      = _model_state["device"]

    # Validar tipo de arquivo
    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(
            status_code=422,
            detail=f"Tipo de arquivo não suportado: '{file.content_type}'. Use JPEG ou PNG.",
        )

    try:
        raw_bytes = await file.read()
        pil_img   = _bytes_to_pil(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao ler imagem: {exc}")

    try:
        tensor, rgb_array = _preprocess(pil_img)
        classe, confianca, probs = _run_inference(model, tensor, class_names, device)
        target_idx = class_names.index(classe)
        heatmap_b64 = _generate_heatmap_b64(model, tensor, rgb_array, target_idx, device)
    except Exception as exc:
        logger.exception("Erro durante inferência")
        raise HTTPException(status_code=500, detail=f"Erro durante inferência: {exc}")

    return PredictResponse(
        doenca_detectada=classe,
        confianca=round(confianca, 6),
        probabilidades={k: round(v, 6) for k, v in probs.items()},
        heatmap_base64=heatmap_b64,
    )


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", summary="Verificação de saúde da API")
def health():
    """Retorna o estado do servidor e se o modelo está carregado."""
    model_loaded = _model_state["model"] is not None
    return {
        "status": "ok",
        "model_loaded": model_loaded,
        "classes": _model_state["class_names"] if model_loaded else None,
        "device": str(_model_state["device"]) if model_loaded else None,
    }


# ── Ponto de entrada ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
