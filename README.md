# 🌽 Corn Analyzer (Dual Model)

> Pipeline completo com **Arquitetura de Modelo Duplo** (Estágios + Pragas) usando **MobileNetV2** + **FastAPI** + **Grad-CAM**, com frontend web dinâmico para análise dupla de imagens.

Este sistema avalia uma única imagem e passa por **duas inteligências artificiais simultâneas**:
1. Uma que classifica em qual estágio de desenvolvimento (ex: V1 a V4) o milho está.
2. Outra que busca identificar pragas ou condições de saúde (ex: Lagarta-do-cartucho).

---

## 📋 Pré-requisitos

| Requisito | Versão mínima |
|-----------|--------------|
| Python | 3.10+ |
| CUDA (opcional) | 11.8+ (para GPU) |
| Espaço em disco | ~3 GB (2 datasets + 2 modelos) |

---

## 🚀 Setup Completo — Passo a Passo

### 1. Instale as dependências
```powershell
pip install -r requirements.txt
# Ou se preferir instalar manualmente:
pip install torch torchvision fastapi uvicorn pydantic python-multipart opencv-python pytorch-grad-cam roboflow python-dotenv google-generativeai
```

---

### 2. Configure as chaves de API
Crie um arquivo `.env` na raiz do projeto contendo:
```
ROBOFLOW_API_KEY=sua_chave_aqui
GEMINI_API_KEY=sua_chave_gemini_aqui
```

---

### 3. Baixe e organize os Datasets (Estágio e Praga)
```powershell
python download_dataset.py
```
O script fará o download de dois datasets do Roboflow (`capstone-maize-growth` e `corn-pest-v4-ybizr`) e organizará tudo nas pastas `dataset/organized_stages` e `dataset/organized_pests`.

---

### 4. Treine os Modelos
```powershell
python train.py
```
O sistema treinará sequencialmente o modelo de Estágios e o modelo de Pragas.
Serão gerados dois arquivos principais:
- `best_model_stages.pth`
- `best_model_pests.pth`
*(Além das curvas de treinamento em PNG).*

> ⚡ GPU CUDA é usada automaticamente se disponível.

---

### 5. Inicie a API Dupla
```powershell
python api.py
```
A API estará disponível em: **http://localhost:8000**
Documentação interativa (Swagger): **http://localhost:8000/docs**

---

### 6. Abra o Frontend
Abra o arquivo `frontend/index.html` diretamente no navegador:
```powershell
start frontend\index.html
```

A interface enviará a imagem para a API e exibirá, lado a lado:
- A classificação do Estágio Fenológico (com mapa de calor e confiança).
- A identificação da Praga/Doença (com mapa de calor e confiança).

---

## 🔌 Endpoints da API

### `POST /predict`
Recebe uma imagem e roda inferência dupla (Stages e Pests).

**Response:**
```json
{
  "estagio": {
    "classe_detectada": "Maize Growth Stage 3",
    "confianca": 0.98,
    "probabilidades": { ... },
    "heatmap_base64": "..."
  },
  "praga": {
    "classe_detectada": "Fall Armyworm",
    "confianca": 0.85,
    "probabilidades": { ... },
    "heatmap_base64": "..."
  }
}
```

### `GET /health`
Verifica se a API e os **dois** modelos estão operacionais e carregados na memória.
