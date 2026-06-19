# Changelog

## [1.2.0] — 2026-06-18

### Correção de Bugs (11 encontrados)
- **11 bugs de runtime corrigidos**: `math`, `random`, `time`, `Any`, `Dict`, `logger`, `BrawlerConfig`, `result`, `cube_pos` — nomes indefinidos que causariam crash
- Todos os `raise ... from` adicionados para preservar stack traces
- `except Exception: pass` substituídos por logging em todos os casos críticos

### Refatoração de Arquitetura
- `core/humanization_utils.py` — extraído de `pylaai_real/`, invertendo a dependência
- `pylaai_real/humanization_utils.py` → shim com `DeprecationWarning`
- `core/polling.py` — `wait_for_condition()`, `AdaptiveSleep` para substituir `time.sleep()`
- `core/handlers/` — handlers de estado como módulos independentes
- `core/handlers/end_game_handler.py` — _handle_end_game extraído (324L → 210L)
- `core/state_transitions.py` — 1404L → 1083L (-321L)
- `core/templates/dashboard.html` — HTML extraído do Python (2185L → 3L)
- `core/dashboard_templates.py` — agora carrega template de arquivo externo
- `__init__.py` criados em 5 pacotes que estavam sem

### Alinhamento de Modelo YOLO
- `config.json` corrigido: modelo padrão agora é `brawlstars_yolov8_8class.pt` (8 classes)
- Modelos órfãos/quebrados movidos para `models/quarantine/`: `bc_unified_best.pt`, `cql_unified_best.pt`, `yolov5s.pt`, `yolov8n_root.pt`
- 4 modelos `.pt` removidos da raiz do projeto
- **Recomendação**: YOLO11 treinado por 11 épocas alcançou mAP50=0.8096 — melhor que YOLOv8n (0.7693)

### Limpeza de Código
- Ruff --fix aplicado em core/ (2032 erros), vision/ (639), decision/ (699), plugins/ (6)
- **Todos os 5 módulos principais: 0 erros ruff**
- 16 arquivos .log movidos para `logs/archive/`
- Caminhos absolutos de usuário removidos de test_full_cycle.py e test_lobby_flow.py

## [1.1.0] — 2026-06-18

### Melhorias Estruturais
- Refatoração completa da raiz: removidos 214 arquivos PNG de debug
- `config.json` populado (estava vazio)
- `src` convertido de arquivo 0 bytes para diretório de código fonte
- `README.md` reescrito com documentação real
- `requirements.txt` sincronizado com `pyproject.toml`
- `main.py` criado como entry point unificado
- `start.bat` criado para inicialização automática

### Refatoração de Arquitetura
- `pylaai_real/` — módulos marcados como deprecados com migration path para `core/`
- Detecção de estado extraída para `core/state_detection.py`
- Transições de estado extraídas para `core/state_transitions.py`

## [1.0.0] — 2026-06-10

- Initial release
