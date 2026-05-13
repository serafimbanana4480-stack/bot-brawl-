# Guia Rápido - Sistema de Treinamento de IA

## Como Começar Rapidamente

### 1. Coletar Dados de Gameplay

```bash
# Grave 30 minutos de gameplay
python -m brawl_bot.automation.gameplay_recorder --adb-id 127.0.0.1:5555 --duration 1800
```

Isso criará:
- `recordings/session_<timestamp>/gameplay.mp4` - Vídeo comprimido
- `recordings/session_<timestamp>/frames/` - Frames individuais
- `recordings/session_<timestamp>/metadata.jsonl` - Metadata rico
- `recordings/session_<timestamp>/actions.jsonl` - Ações registradas

### 2. Organizar Dataset para YOLO

```bash
# Mova frames para dataset
mkdir -p dataset/raw/images
cp recordings/session_*/frames/*.png dataset/raw/images/

# Organize dataset
python -m brawl_bot.dataset_pipeline --adb-id 127.0.0.1:5555 --duration 60
```

### 3. Treinar Modelo YOLO

```bash
# Treinamento simples
python -m brawl_bot.training.train_brawlstars --data ./dataset --epochs 100 --device cpu

# Treinamento progressivo (recomendado)
python -m brawl_bot.training.train_brawlstars --data ./dataset --epochs 100 --progressive --device cpu
```

### 4. Coletar Dados para Behavior Cloning

```bash
# Grave gameplay humano com ações
python -m brawl_bot.automation.gameplay_recorder --adb-id 127.0.0.1:5555 --duration 600

# Anote manualmente ou use auto-labeling
```

### 5. Treinar Modelo de Behavior Cloning

```python
from rl_stubs.behavior_cloning import BehaviorCloningTrainer, BCConfig
from pathlib import Path

# Configurar
config = BCConfig(
    dataset_dir=Path("dataset/labeled"),
    output_model_path=Path("models/bc_policy.pt"),
    epochs=50,
    batch_size=32,
    device="cpu"
)

# Treinar
trainer = BehaviorCloningTrainer(config)
result = trainer.train()

print(f"Treinamento concluído: {result}")
```

### 6. Usar Modelo BC no Bot

```python
from rl_stubs.behavior_cloning import BehaviorCloningTrainer, BCConfig
import cv2

# Carregar modelo
config = BCConfig()
trainer = BehaviorCloningTrainer(config)
trainer.load_policy(Path("models/bc_policy.pt"))

# Capturar frame
image = cv2.imread("frame.png")
aux_state = np.array([1.0, 3.0])  # [health, ammo]

# Prever ação
action = trainer.predict(image, aux_state)
print(f"Ação prevista: {action}")
```

## Auto-Labeling com SAM2

### Instalar SAM2

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Usar Auto-Labeling

```python
from training.sam2.sam2_auto_labeler import SAM2AutoLabeler
import cv2

# Inicializar
labeler = SAM2AutoLabeler(model_type="sam2_hiera_small")

# Carregar vídeo
video_frames = []  # Carregar frames do vídeo
seed_labels = {}  # Adicionar 20-30 frames anotados manualmente

# Propagar labels
propagated = labeler.propagate_labels(video_frames, seed_labels)

# Exportar para YOLO
labeler.export_yolo_dataset(
    video_frames,
    propagated,
    Path("dataset_sam2"),
    ["enemy", "teammate", "player", "wall", "bush", "powerup", "box", "bullet"]
)
```

## Validação de Modelos

### Validar YOLO

```bash
python -m brawl_bot.training.validate_and_register_model
```

### Benchmark Modelos

```python
from training.yolo11.benchmark_models import benchmark_models

results = benchmark_models(
    models_dir=Path("models"),
    test_images=Path("dataset/val/images")
)
print(results)
```

## Troubleshooting

### Erro: SAM2 não instalado

```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

### Erro: CUDA out of memory

Use CPU em vez de GPU:
```bash
python -m brawl_bot.training.train_brawlstars --data ./dataset --device cpu
```

### Erro: Dataset vazio

Verifique se o dataset existe:
```bash
ls dataset/raw/images/
ls dataset/train/images/
```

### Erro: PyTorch não instalado

```bash
pip install torch torchvision
```

## Próximos Passos

1. Coletar mais dados (mínimo 1000 frames por classe para YOLO)
2. Anotar manualmente 20-30 frames para SAM2
3. Treinar YOLO com dataset real
4. Coletar 10.000+ transições para BC
5. Treinar BC
6. Integrar BC no sistema de decisão
7. Implementar CQL (próxima fase)

## Suporte

Para mais informações, consulte:
- `IMPLEMENTATION_PROGRESS.md` - Progresso detalhado
- `ARCHITECTURE.md` - Arquitetura do sistema
- `FEATURES_DOCUMENTATION.md` - Features existentes
