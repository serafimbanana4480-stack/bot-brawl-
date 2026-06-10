# Data Collection Mode — GameplayCollector

## Overview

O `GameplayCollector` captura dados completos de cada frame durante o gameplay para treinar a `NeuralPolicy` via PPO. Cada episódio (partida) é guardado em `dataset/raw/episodes/episode_XXXX/` com:

- `metadata.json` — brawler, mapa, resultado, estatísticas RL
- `frames/` — um JSON por frame com `state_vector` (44-dim), `action_idx`, `reward`, `next_state_vector`, `log_prob`, `value`, `grid` (21×21), `done`
- `screenshots/` — screenshots JPEG por frame (opcional, para debug)

## Configuração

Editar `config.json`:

```json
{
  "rl": {
    "enabled": true,
    "data_collection_mode": true,
    "collect_screenshots": true,
    "collect_grids": true,
    "reward_shaping": {
      "damage_dealt": 0.1,
      "damage_taken": -0.1,
      "power_cube_collected": 0.05,
      "kill": 0.2,
      "win": 1.0,
      "loss": -1.0,
      "timestep_penalty": -0.001,
      "survival_time": 0.05,
      "survival_interval": 5.0,
      "reward_clip": 1.0,
      "normalize": true
    }
  }
}
```

- `data_collection_mode: true` — ativa coleta de dados RL completos
- `collect_screenshots: true` — guarda screenshots (ocupa mais disco)
- `collect_grids: true` — guarda spatial grids 21×21

## Modos de Operação

### 1. Modo "Data Collection" (coleta dados, não treina)

Neste modo o bot joga normalmente mas guarda todos os dados RL. O treino PPO pode ser feito offline depois.

```python
from dataset.collector import GameplayCollector
from pylaai_real.rl_engine import OnlineLearner
from core.reward_bridge import RewardBridge

collector = GameplayCollector(
    base_dir="dataset/raw",
    collect_screenshots=True,
    collect_grids=True,
)
reward_bridge = RewardBridge()
learner = OnlineLearner(
    reward_bridge=reward_bridge,
    gameplay_collector=collector,
    use_neural=True,
)

# O loop de jogo chama automaticamente:
#   learner.start_episode("colt", "map_name")
#   action, confidence = learner.get_action(state, ...)
#   learner.learn_from_frame(state, action, reward, next_state, detections=...)
#   learner.end_episode(result="win", rank=1)
```

### 2. Modo "Play" (apenas joga, não coleta)

```python
learner = OnlineLearner(
    reward_bridge=reward_bridge,
    gameplay_collector=None,  # desativa coleta
    use_neural=True,
)
```

### 3. Modo "Play + Online Learning" (coleta + treina online)

O `RLBridge` já treina PPO online a cada `train_every_n_steps` frames. Basta manter `use_neural=True` e o `ExperienceBuffer` acumula transições.

## Estrutura de um Episódio

```
dataset/raw/episodes/episode_0001/
├── metadata.json
├── frames/
│   ├── frame_00001.json
│   ├── frame_00002.json
│   └── ...
└── screenshots/
    ├── frame_00001.jpg
    ├── frame_00002.jpg
    └── ...
```

Exemplo de `frame_00001.json`:

```json
{
  "timestamp": 1778629963.69,
  "frame_id": 1,
  "state": "in_game",
  "detections": {
    "player": [[100, 200, 150, 250]],
    "enemy": [[400, 300, 450, 350]]
  },
  "action": {"name": "attack", "confidence": 0.85},
  "reward": 0.05,
  "state_vector": [0.8, 1.0, 0.0, ...],
  "action_idx": 1,
  "next_state_vector": [0.75, 0.9, 0.1, ...],
  "log_prob": -0.42,
  "value": 1.23,
  "grid": [[[0, 0, 1, ...], ...], ...],
  "done": false
}
```

## Validação

Após coletar episódios, correr o script de validação:

```bash
# Windows (Git Bash / PowerShell)
python scripts/validate_episodes.py

# Verificar e remover episódios vazios
python scripts/validate_episodes.py --fix-empty

# Especificar diretório custom
python scripts/validate_episodes.py --episodes-dir dataset/raw/episodes
```

O script verifica:
- Episódio tem >100 frames
- Cada frame tem `state_vector`, `action_idx`, `reward`, `next_state_vector`
- Rewards não são todos 0.0
- Deteções YOLO presentes

O relatório é guardado em `dataset/raw/validation_report.json`.

## Integração com o Orchestrator

O `BotOrchestrator` (core/orchestrator.py) já chama `decision.start_episode()` e `decision.end_episode()` nas transições de estado. Para ativar a coleta de dados, basta injetar um `OnlineLearner` com `gameplay_collector` no `DecisionPort` adapter.

Exemplo no `core/factory.py` (ou onde o orchestrator é criado):

```python
from dataset.collector import GameplayCollector
from pylaai_real.rl_engine import OnlineLearner
from core.reward_bridge import RewardBridge

collector = GameplayCollector()
reward_bridge = RewardBridge(data_collector=collector)
learner = OnlineLearner(
    reward_bridge=reward_bridge,
    gameplay_collector=collector,
    use_neural=True,
)
# Inject learner no DecisionPort adapter
```

## Troubleshooting

### Episódios vazios (0 frames)
- Verificar se `learn_from_frame()` está sendo chamado no loop principal
- Verificar se `start_episode()` é chamado antes do primeiro frame

### Rewards todos 0.0
- Verificar se `RewardBridge.log_combat_frame()` está sendo chamado
- Verificar se `reward_shaping` está configurado em `config.json`
- Verificar se o `Orchestrator` está a passar `damage_dealt`, `damage_taken`, etc. para `learn_from_frame()`

### Deteções null
- Verificar se o `VisionPort` está a retornar `detected_objects`
- Verificar se o `YOLOVisionAdapter` está carregado e a detetar objetos

### Disco cheio
- Desativar `collect_screenshots: false` em `config.json`
- Rodar `python scripts/validate_episodes.py --fix-empty` para limpar episódios vazios
