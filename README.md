# 🐄 Cattle Disease Classifier

> Pipeline completo de detecção de doenças bovinas usando **MobileNetV2** + **FastAPI** + **Grad-CAM**, com frontend web para análise de imagens.

---

## 📋 Pré-requisitos

| Requisito | Versão mínima |
|-----------|--------------|
| Python | 3.10+ |
| CUDA (opcional) | 11.8+ (para GPU) |
| Espaço em disco | ~2 GB (dataset + modelo) |

---

## 🚀 Setup Completo — Passo a Passo

### 1. Clone ou acesse o projeto

```powershell
cd "caminho\para\teste de ideia"
```

---

### 2. Crie e ative um ambiente virtual *(recomendado)*

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **Nota:** Se o PowerShell bloquear a execução de scripts, rode antes:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

---

### 3. Instale as dependências

```powershell
python setup.py
```

Isso instala automaticamente: PyTorch, torchvision, FastAPI, Grad-CAM, OpenCV, Roboflow SDK e demais pacotes.

---

### 4. Configure a chave do Roboflow

A chave já está configurada no arquivo `.env` do projeto. Caso precise substituí-la, edite o arquivo `.env`:

```
ROBOFLOW_API_KEY=sua_chave_aqui
```

Ou obtenha uma nova chave em [roboflow.com → Settings → API](https://app.roboflow.com/).

---

### 5. Baixe e organize o dataset

```powershell
python download_dataset.py
```

O script irá:
- Baixar o dataset `cattle-diseases` do Roboflow
- Reorganizar em splits **70% train / 20% val / 10% test**
- Imprimir uma tabela com o número de imagens por classe

---

### 6. Treine o modelo

```powershell
python train.py
```

Após o treino, serão gerados:
- `best_model.pth` — melhor checkpoint salvo automaticamente
- `training_curves.png` — gráfico de acurácia/loss por época

> ⚡ GPU CUDA é usada automaticamente se disponível.

---

### 7. Inicie a API

```powershell
python api.py
```

Ou com recarregamento automático (modo desenvolvimento):

```powershell
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

A API estará disponível em: **http://localhost:8000**
Documentação interativa (Swagger): **http://localhost:8000/docs**

---

### 8. Abra o Frontend

Abra o arquivo `frontend/index.html` diretamente no navegador:

```powershell
start frontend\index.html
```

A interface permite enviar imagens e visualizar:
- Classe detectada e nível de confiança
- Probabilidades por doença em gráfico de barras
- Mapa de calor Grad-CAM sobreposto à imagem original

---

## 📁 Estrutura do Projeto

```
teste de ideia/
├── .env                        ← Variáveis de ambiente (API Key)
├── setup.py                    ← Instala dependências
├── download_dataset.py         ← Baixa e organiza o dataset
├── train.py                    ← Fine-tuning do MobileNetV2
├── retrain.py                  ← Re-treino com dataset limpo
├── clean_dataset.py            ← Limpeza e filtragem do dataset
├── gradcam_inference.py        ← Inferência local com Grad-CAM
├── compare_models.py           ← Comparação entre modelos
├── api.py                      ← Servidor FastAPI (REST API)
├── best_model.pth              ← Modelo treinado (gerado pelo train.py)
├── best_model_clean.pth        ← Modelo treinado com dataset limpo
├── training_curves.png         ← Gráfico de curvas de treino
├── frontend/
│   └── index.html              ← Interface web para análise de imagens
└── dataset/
    └── organized/
        ├── train/
        ├── val/
        └── test/
```

---

## 🔌 Endpoints da API

### `POST /predict`

Recebe uma imagem e retorna a doença detectada com mapa de calor.

**Request:** `multipart/form-data` com campo `file` (JPG, PNG, WEBP ou BMP)

**Response:**
```json
{
  "doenca_detectada": "lumpy_skin_disease",
  "confianca": 0.912345,
  "probabilidades": {
    "lumpy_skin_disease": 0.912345,
    "foot_and_mouth": 0.071234,
    "healthy": 0.016421
  },
  "heatmap_base64": "<JPEG em base64>"
}
```

### `GET /health`

Verifica se a API e o modelo estão operacionais.

---

## 🧪 Teste via cURL

```bash
curl -X POST http://localhost:8000/predict \
  -F "file=@caminho/para/imagem.jpg" \
  | python -m json.tool
```

---

## ⚙️ Parâmetros Configuráveis

| Arquivo | Parâmetro | Padrão | Descrição |
|---------|-----------|--------|-----------|
| `train.py` | `EPOCHS` | 20 | Máximo de épocas |
| `train.py` | `BATCH_SIZE` | 32 | Tamanho do batch |
| `train.py` | `LR` | 1e-4 | Learning rate inicial |
| `train.py` | `PATIENCE_STOP` | 5 | Early stopping (épocas) |
| `download_dataset.py` | `VERSION` | 1 | Versão do dataset no Roboflow |
| `api.py` | `port` | 8000 | Porta do servidor |

---

## 🩺 Inferência Local (sem API)

```powershell
python gradcam_inference.py caminho\para\imagem.jpg
```

Salva o resultado como `resultado_gradcam.jpg` na pasta do projeto.

---

## ❓ Troubleshooting

**"Modelo não carregado" (503):**
> Verifique se o arquivo `best_model.pth` existe. Se não, execute `python train.py` primeiro.

**Erro ao instalar PyTorch com CUDA:**
> Instale manualmente seguindo as instruções em [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

**ExecutionPolicy no PowerShell:**
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

**CORS bloqueado no frontend:**
> Certifique-se de que a API está rodando em `http://localhost:8000`. O CORS já está configurado para aceitar qualquer origem.
