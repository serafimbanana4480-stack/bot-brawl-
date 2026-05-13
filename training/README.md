# Brawl Stars Dataset Collection & Training Pipeline

Este documento descreve o pipeline completo para coleta de dataset e treinamento de modelo YOLOv8 para o Brawl Stars Bot.

## Overview

O pipeline consiste em 4 etapas principais:

1. **Coleta de Dataset** - Captura automatizada de screenshots durante gameplay real
2. **Auto-Labeling** - Rotulagem automática usando SAM2 (Segment Anything Model 2)
3. **Treinamento YOLO** - Treinamento do modelo YOLOv8 com o dataset coletado
4. **Validação e Registro** - Validação do modelo e integração com o sistema

## Pré-requisitos

- Python 3.8+
- Emulador Android (BlueStacks, LDPlayer, etc.) rodando Brawl Stars
- ADB configurado e conectado ao emulador
- Dependências instaladas (ver `requirements_training.txt`)

## Instalação

```bash
cd "ia ultima/soberana-omega/backend"
pip install -r requirements_training.txt
```

## Etapa 1: Coleta de Dataset

### Captura Contínua (Recomendado)

Capture screenshots continuamente durante gameplay por um período específico:

```bash
python -m brawl_bot.automation.dataset_collector \
    --duration 300 \
    --interval 2.0 \
    --output ./dataset/raw
```

Parâmetros:
- `--duration`: Duração em segundos (default: 300 = 5 minutos)
- `--interval`: Intervalo entre captures em segundos (default: 2.0)
- `--output`: Diretório de saída (default: ./dataset/raw)

### Captura Baseada em Eventos

Capture screenshots baseado em mudanças de estado do jogo:

```bash
python -m brawl_bot.automation.dataset_collector \
    --mode event \
    --max-frames 1000 \
    --output ./dataset/raw
```

Parâmetros:
- `--mode`: Modo de coleta (continuous ou event)
- `--max-frames`: Máximo de frames (modo event only)

### Estrutura de Saída

O coletor criará a seguinte estrutura:

```
dataset/raw/
├── raw/                          # Screenshots brutos
│   ├── 20260509_143025_001.png
│   ├── 20260509_143027_002.png
│   └── ...
├── by_state/                     # Organizado por estado do jogo
│   ├── lobby/
│   ├── matchmaking/
│   ├── game/
│   └── ...
├── metadata/                     # Metadados JSON para cada frame
│   ├── 20260509_143025_001.json
│   └── ...
└── collection_report.json        # Relatório da coleta
```

## Etapa 2: Auto-Labeling com SAM2

### Rotulagem Manual (Seed Frames)

Para iniciar o auto-labeling, você precisa rotular manualmente 20-30 frames (seed frames):

1. Use uma ferramenta de rotulagem como Label Studio ou CVAT
2. Rotule as seguintes classes:
   - `enemy`: Brawlers inimigos
   - `teammate`: Brawlers aliados
   - `player`: Seu brawler
   - `wall`: Paredes e obstáculos
   - `bush`: Arbustos
   - `powerup`: Power-ups e itens
   - `box`: Caixas e baús
   - `bullet`: Projéteis

3. Salve labels no formato YOLO (`.txt` com coordenadas normalizadas)

### Auto-Labeling com SAM2

Use o auto-labeler para propagar as labels para o restante do dataset:

```bash
python -m brawl_bot.training.auto_labeler_v2 \
    --input ./dataset/raw/raw \
    --seed-labels ./dataset/raw/seed_labels \
    --output ./dataset/labeled
```

Parâmetros:
- `--input`: Diretório com imagens não rotuladas
- `--seed-labels`: Diretório com labels manuais (seed)
- `--output`: Diretório de saída para labels propagados

O auto-labeler usará SAM2 para segmentar objetos e propagar as labels baseadas nos seed frames.

## Etapa 3: Treinamento YOLO

### Preparar Dataset

O script de treinamento organizará automaticamente o dataset em train/val splits:

```bash
python -m brawl_bot.training.train_brawlstars \
    --data ./dataset/labeled \
    --epochs 100 \
    --model yolov8n.pt \
    --output ./models
```

Parâmetros:
- `--data`: Diretório do dataset (deve conter images/ e labels/)
- `--epochs`: Número de épocas (default: 100)
- `--model`: Modelo base (default: yolov8n.pt)
- `--output`: Diretório de saída (default: ./models)
- `--pretrained`: Usar pesos pré-treinados (default: True)
- `--batch-size`: Batch size (default: 16)
- `--img-size`: Image size (default: 640)
- `--device`: Device para treinamento (default: cpu, use 0 para GPU)

### Treinamento do Zero (Sem Pre-training)

Para treinar do zero sem usar pesos COCO:

```bash
python -m brawl_bot.training.train_brawlstars \
    --data ./dataset/labeled \
    --epochs 100 \
    --model yolov8n.pt \
    --pretrained=False
```

### Treinamento com GPU

Se você tiver GPU disponível:

```bash
python -m brawl_bot.training.train_brawlstars \
    --data ./dataset/labeled \
    --epochs 100 \
    --device 0
```

## Etapa 4: Validação e Registro

Após o treinamento, valide e registre o modelo no sistema:

```bash
python -m brawl_bot.training.validate_and_register_model \
    --model ./models/brawlstars/weights/best.pt \
    --config config.json
```

Este script irá:
1. Inspecionar o modelo treinado
2. Validar se é adequado para Brawl Stars
3. Registrar no `model_registry.json`
4. Atualizar `config.json` para usar o novo modelo

## Workflow Completo (Exemplo)

```bash
# 1. Coletar dataset (5 minutos de gameplay)
python -m brawl_bot.automation.dataset_collector \
    --duration 300 \
    --output ./dataset/raw

# 2. Rotular manualmente 20-30 frames (seed)
# (Use Label Studio ou CVAT para rotular ./dataset/raw/raw)

# 3. Auto-labeling com SAM2
python -m brawl_bot.training.auto_labeler_v2 \
    --input ./dataset/raw/raw \
    --seed-labels ./dataset/raw/seed_labels \
    --output ./dataset/labeled

# 4. Treinar modelo YOLO
python -m brawl_bot.training.train_brawlstars \
    --data ./dataset/labeled \
    --epochs 100 \
    --output ./models

# 5. Validar e registrar modelo
python -m brawl_bot.training.validate_and_register_model \
    --model ./models/brawlstars/weights/best.pt \
    --config config.json
```

## Classes do Dataset

O dataset de Brawl Stars usa as seguintes classes:

| ID | Classe | Descrição |
|----|--------|-----------|
| 0  | enemy  | Brawlers inimigos |
| 1  | teammate | Brawlers aliados |
| 2  | player | Seu brawler (personagem controlado) |
| 3  | wall  | Paredes e obstáculos |
| 4  | bush  | Arbustos (esconderijo) |
| 5  | powerup | Power-ups e itens coletáveis |
| 6  | box   | Caixas e baús |
| 7  | bullet | Projéteis e ataques |

## Troubleshooting

### Emulador não detectado

Certifique-se de que:
- O emulador está rodando
- ADB está instalado e configurado
- A porta ADB está correta (default: 127.0.0.1:5555)

Verifique com:
```bash
adb devices
```

### Erro ao carregar modelo

Certifique-se de que:
- Ultralytics está instalado: `pip install ultralytics`
- O modelo existe no caminho especificado
- O modelo não está corrompido

### Validação falhou

Se a validação falhar com "modelo contém classes COCO":
- O modelo não foi treinado com dataset real de Brawl Stars
- Treine novamente com dataset coletado e rotulado

## Próximos Passos

Após registrar o modelo:

1. Teste o modelo no jogo:
   ```bash
   python -m brawl_bot.main --mode auto
   ```

2. Monitore performance e ajuste conforme necessário

3. Colete mais dados se o modelo não estiver performando bem

4. Itere no processo de coleta -> rotulagem -> treinamento

## Referências

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [SAM2 Documentation](https://github.com/facebookresearch/segment-anything-2)
- [Label Studio](https://labelstud.io/)
- [CVAT](https://opencv.github.io/cvat/)
