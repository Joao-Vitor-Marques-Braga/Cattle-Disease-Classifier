"""
api.py — Servidor FastAPI para inferência de estágios e pragas do milho.

Uso:
    python api.py
    # ou com uvicorn diretamente:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from dotenv import load_dotenv
load_dotenv()

import base64
import io
import os
import logging
import google.generativeai as genai
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict

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
IMAGE_SIZE  = 224
MEAN        = [0.485, 0.456, 0.406]
STD         = [0.229, 0.224, 0.225]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Estado global dos modelos ─────────────────────────────────────────────────
_models: Dict[str, Any] = {
    "stages": {"model": None, "class_names": None},
    "pests": {"model": None, "class_names": None},
    "device": None,
}

# ── Schemas de resposta ───────────────────────────────────────────────────────
class PredictionDetail(BaseModel):
    classe_detectada: str
    confianca: float
    probabilidades: dict[str, float]
    heatmap_base64: str

class PredictResponse(BaseModel):
    estagio: PredictionDetail
    praga: PredictionDetail

class LLMReportRequest(BaseModel):
    estagio: str
    praga: str

class LLMReportResponse(BaseModel):
    report_markdown: str


# ── Ciclo de vida: carrega modelos ao iniciar o servidor ──────────────────────
def load_single_model(name: str, device: torch.device):
    path = Path(f"best_model_{name}.pth")
    if not path.exists():
        logger.warning(f"AVISO: Modelo {name} não encontrado em '{path}'.")
        return None, None

    try:
        checkpoint = torch.load(path, map_location=device)
        class_names = checkpoint["class_names"]
        num_classes = checkpoint["num_classes"]

        model = models.mobilenet_v2(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(in_feat, num_classes),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        
        logger.info(f"Modelo {name} carregado — {num_classes} classes.")
        return model, class_names
    except Exception as exc:
        logger.error(f"Falha ao carregar o modelo {name}: {exc}")
        return None, None

@asynccontextmanager
async def lifespan(app: FastAPI):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Dispositivo de inferência: {device}")
    _models["device"] = device

    model_s, classes_s = load_single_model("stages", device)
    _models["stages"] = {"model": model_s, "class_names": classes_s}

    model_p, classes_p = load_single_model("pests", device)
    _models["pests"] = {"model": model_p, "class_names": classes_p}

    yield

    _models["stages"]["model"] = None
    _models["pests"]["model"] = None


# ── Criação da aplicação ─────────────────────────────────────────────────────
app = FastAPI(
    title="API de Análise de Milho (Dual Model)",
    description="Classifica estágios de desenvolvimento e pragas simultaneamente.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Utilitários de processamento ─────────────────────────────────────────────
def _bytes_to_pil(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data)).convert("RGB")

def _preprocess(pil_img: Image.Image):
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
    return tf_norm(pil_img).unsqueeze(0), tf_raw(pil_img).permute(1, 2, 0).numpy()


def _run_inference(model: torch.nn.Module, tensor: torch.Tensor, class_names: list[str], device: torch.device):
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = F.softmax(logits, dim=1).squeeze()
    
    # Se probs for um escalar (apenas 1 classe, improvável mas previne erro)
    if probs.dim() == 0:
        probs = probs.unsqueeze(0)

    idx = probs.argmax().item()
    all_probs = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
    return class_names[idx], float(probs[idx]), all_probs


def _generate_heatmap_b64(model: torch.nn.Module, tensor: torch.Tensor, rgb_array: np.ndarray, target_idx: int, device: torch.device) -> str:
    target_layers = [model.features[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    targets = [ClassifierOutputTarget(target_idx)]

    grayscale_cam = cam(input_tensor=tensor.to(device), targets=targets)[0]
    rgb_normalized = np.clip(rgb_array, 0.0, 1.0).astype(np.float32)
    visualization = show_cam_on_image(rgb_normalized, grayscale_cam, use_rgb=True)

    _, buffer = cv2.imencode(".jpg", cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buffer).decode("utf-8")


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictResponse, summary="Análise Dupla (Estágio e Praga)")
async def predict(file: UploadFile = File(...)):
    if _models["stages"]["model"] is None or _models["pests"]["model"] is None:
        raise HTTPException(status_code=503, detail="Modelos não carregados. Execute o treinamento para stages e pests.")

    if file.content_type not in ("image/jpeg", "image/png", "image/webp", "image/bmp"):
        raise HTTPException(status_code=422, detail="Tipo de arquivo não suportado.")

    try:
        raw_bytes = await file.read()
        pil_img = _bytes_to_pil(raw_bytes)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Erro ao ler imagem: {exc}")

    device = _models["device"]
    try:
        tensor, rgb_array = _preprocess(pil_img)

        # Inferência: Estágios
        m_stages, c_stages = _models["stages"]["model"], _models["stages"]["class_names"]
        stg_cls, stg_conf, stg_probs = _run_inference(m_stages, tensor, c_stages, device)
        stg_hm = _generate_heatmap_b64(m_stages, tensor, rgb_array, c_stages.index(stg_cls), device)

        # Inferência: Pragas
        m_pests, c_pests = _models["pests"]["model"], _models["pests"]["class_names"]
        pst_cls, pst_conf, pst_probs = _run_inference(m_pests, tensor, c_pests, device)
        pst_hm = _generate_heatmap_b64(m_pests, tensor, rgb_array, c_pests.index(pst_cls), device)

    except Exception as exc:
        logger.exception("Erro durante inferência dupla")
        raise HTTPException(status_code=500, detail=f"Erro durante inferência: {exc}")

    return PredictResponse(
        estagio=PredictionDetail(
            classe_detectada=stg_cls,
            confianca=round(stg_conf, 6),
            probabilidades={k: round(v, 6) for k, v in stg_probs.items()},
            heatmap_base64=stg_hm,
        ),
        praga=PredictionDetail(
            classe_detectada=pst_cls,
            confianca=round(pst_conf, 6),
            probabilidades={k: round(v, 6) for k, v in pst_probs.items()},
            heatmap_base64=pst_hm,
        )
    )

@app.post("/generate_llm_report", response_model=LLMReportResponse, summary="Gera laudo agronômico com Gemini")
async def generate_llm_report(req: LLMReportRequest):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY não encontrada no servidor.")
    
    genai.configure(api_key=api_key)
    
    prompt = f"""
Você é um Engenheiro Agrônomo experiente especializado na cultura do milho.
Uma lavoura foi diagnosticada pela visão computacional com o seguinte quadro:
- Estágio de desenvolvimento: {req.estagio}
- Praga ou condição detectada: {req.praga}

Escreva um laudo técnico curto e direto (no máximo 2 a 3 parágrafos curtos) focado em ações práticas para o produtor.
Inclua:
1. Uma breve análise da gravidade dessa praga nesse estágio de crescimento específico.
2. Recomendações imediatas de manejo, controle ou monitoramento.

Responda em formato Markdown, usando negrito para destacar os pontos importantes. 
Não inclua introduções longas como "Claro, aqui está o laudo". Comece direto no assunto.
""".strip()
    
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return LLMReportResponse(report_markdown=response.text)
    except Exception as exc:
        logger.exception("Erro ao gerar relatório com Gemini")
        raise HTTPException(status_code=500, detail=f"Erro ao gerar laudo com IA: {exc}")

@app.get("/health")
def health():
    return {
        "status": "ok",
        "stages_loaded": _models["stages"]["model"] is not None,
        "pests_loaded": _models["pests"]["model"] is not None,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
