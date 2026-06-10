# Relatório de Correções — Data Collection & Testes

## Resumo Executivo

Foram corrigidos **4 testes falhados** e melhorado o sistema de data collection com um novo script standalone. Todos os testes do projeto passam agora (**923 passed, 39 skipped**).

---

## 1. Testes Corrigidos

### 1.1 `tests/test_orchestrator_integration.py`
**Teste:** `TestHappyPath::test_match_persists_and_actions_execute`
**Problema:** O `RepetitionGuard` tem um cooldown default de `2.0s` entre ações idênticas. Como o teste usava o mesmo `target_pos=(0.5, 0.5)` para 3 ticks consecutivos (executados em <1ms cada), o guard vetou as ações 2 e 3, resultando em `actions_executed=1` em vez de `>=3`.
**Correção:** O teste agora fornece uma queue de `Decision` com targets distintos `(0.45,0.45)`, `(0.46,0.46)`, `(0.47,0.47)` no `FakeDecision`, evitando o cooldown.

### 1.2 `tests/test_strategic_improvements.py`
**Testes:** `TestDegradationManager::test_degradation_on_errors` e `test_recovery`
**Problema:** O `DegradationManager` calcula a taxa de erro como `errors_last_min / len(recent_errors)`. Com 25 erros consecutivos e zero sucessos, a taxa é **100%**, o que dispara imediatamente o modo `EMERGENCY` (threshold default 0.80) ou `MINIMAL` (threshold default 0.60), saltando o `DEGRADED` esperado pelo teste.
**Correção:** Ambos os testes agora instanciam `DegradationManager` com `error_threshold_minimal=2.0` e `error_threshold_emergency=2.0`, desativando os modos superiores e permitindo testar `DEGRADED` isoladamente.

### 1.3 `tests/test_wrapper_diagnostics_status.py`
**Teste:** `test_get_status_exposes_diagnostics_snapshot`
**Problema:** O `get_status()` em `wrapper.py` tentava chamar `state_manager.get_current_state_name()` diretamente, mas o método real está em `state_manager.screen_automation.get_current_state_name()`. O stub do teste só tinha `screen_automation` como atributo, não o método no `state_manager`.
**Correção:** Linha 742 de `wrapper.py` ajustada para:
```python
"screen_state": (
    _safe_get("state_manager", "screen_automation", None).get_current_state_name()
    if _safe_get("state_manager", "screen_automation", None) is not None
    else "unknown"
),
```

### 1.4 `tests/test_mlflow_tracking.py`
**Teste:** `TestRLBridgeWarmStart::test_warm_start_loads_latest`
**Problema:** Uma tentativa anterior de patch em `neural.rl_bridge.NeuralPolicy` falhou porque o `RLBridge` faz lazy import de `neural.neural_policy.NeuralPolicy` dentro do `__init__`.
**Correção:** Revertido para o patch correto `neural.neural_policy.NeuralPolicy`.

---

## 2. Melhorias no Sistema de Data Collection

### 2.1 Script standalone criado
**Ficheiro:** `scripts/run_data_collection.py`
**Funcionalidades:**
- Inicializa a stack completa `OnlineLearner + GameplayCollector + RewardBridge`
- Verifica/ativa `rl.data_collection_mode` no `config.json`
- Corre `N` partidas simuladas (configurável via `--matches`)
- Valida cada episódio após finalização (verifica `metadata.json`, frame count, screenshots)
- Gera relatório JSON em `dataset/raw/collection_report.json`
- Suporta `--dry-run` para testes rápidos

**Uso:**
```bash
.venv/Scripts/python.exe scripts/run_data_collection.py --matches 10
```

### 2.2 Configuração verificada
O `config.json` já contém:
```json
"rl": {
  "data_collection_mode": true,
  "collect_screenshots": true,
  "collect_grids": true
}
```

### 2.3 Integração no loop verificada
- **`OnlineLearner.learn_from_frame()`** chama `GameplayCollector.record_frame()` corretamente (linhas 499-534 de `pylaai_real/rl_engine.py`)
- **`RewardBridge`** é conectado ao loop de gameplay em `core/state_transitions.py::_handle_in_game()` (linhas 820-827)
- **`ExperienceBuffer`** recebe dados via `RLBridge` quando `use_neural=True`

---

## 3. Verificação End-to-End

Teste de integração manual executado com sucesso:
```python
from pylaai_real.rl_engine import OnlineLearner
from dataset.collector import GameplayCollector
from core.reward_bridge import RewardBridge
from core.experience_buffer import ExperienceBuffer

# Inicialização OK
# Start episode OK
# 10 frames de learn_from_frame OK
# End episode OK
# Collector stats: {'total_episodes': 1, 'total_frames': 11}
# Buffer size: 10
# Reward summary com total_reward, normalized_reward, etc.
```

**Resultado:** Sem erros de import circular. Todos os componentes comunicam corretamente.

---

## 4. Resultado Final dos Testes

```
platform win32 -- Python 3.12.10, pytest-9.0.3
923 passed, 39 skipped, 9 warnings in 98.98s
```

Todos os testes do projeto estão verdes.
