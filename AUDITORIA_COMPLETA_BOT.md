# Auditoria Completa do Bot Brawl Stars

**Data:** 2026-05-14
**Escopo:** Código morto, utilidade real do bot, integridade da dashboard, funcionalidades fantasma.

---

## 1. Resumo Executivo

| Categoria | Estado | Notas |
|-----------|--------|-------|
| Loop principal do bot | **Funcional** | Captura screenshots, deteta estados, executa combat |
| Combat / PlayLogic | **Funcional** | Usa YOLO deteções + envia inputs ADB |
| Dashboard | **Zero mock confirmado** | Todos os dados vêm do wrapper em execução |
| Código morto | **Crítico** | 70+ módulos não importados por ninguém |
| Funcionalidades fantasma | **Presentes** | Wall detection, OCR mapas, HP real não implementados |

---

## 2. Código Morto / Não Utilizado

### 2.1 Módulos nunca importados (70+)

Estes módulos existem no repositório mas **nenhum outro ficheiro os importa**:

- `api/` — `api.py`, `api/brawl_stars_routes.py` (API REST/FastAPI não ligada ao wrapper)
- `automation/` — `bot_monitoring.py`, `code_quality.py`, `dataset_collector.py`, `dataset_collector_v2.py`, `emulator_monitoring.py`, `model_download.py`, `rule_engine.py`, `gameplay_recorder.py`
- `control/` — módulo vazio
- `dataset/create_sample_datasets.py`
- `decision/map_strategy.py`, `decision/neural_policy.py`
- `enterprise/` — **toda a pasta** (74 itens: agents, memory, vision, simulation, API, etc.) — zero imports pelo wrapper
- `install_training_deps.py`
- `lobby_runner.py` — entry point legado não usado pelo wrapper principal
- `main.py`, `main_enhanced.py` — entry points legados
- `pylaai_real/anti_ban_advanced.py`, `elo_tracker.py`, `humanization_utils.py`
- `rl_stubs/safety_gate.py`
- `setup_wizard.py`
- `vision/map_analyzer.py`
- Vários `test_*.py` na raiz (`test_brawl_stars_integration.py`, `test_improvements.py`, `test_lobby_live.py`, etc.)
- Grande parte de `training/` — `auto_labeler.py`, `auto_labeler_v2.py`, `complete_functionality_test.py`, `continuous_learning_loop.py`, `dataset_manager.py`, `extended_training_test.py`, `label_studio/*`, `massive_synthetic_generator.py`, `professional_training_pipeline.py`, `sam2/*`, `simple_rewards_test.py`, `test_learning_system.py`, `test_rewards_comprehensive.py`, `train_ai.py`, `train_behavior_cloning.py`, `train_brawlstars.py`, `train_cql.py`, `train_with_massive_data.py`, `train_yolo.py`, `validate_and_register_model.py`, `yolo11/benchmark_models.py`, `yolo_world/*`

### 2.2 Funções com corpo vazio (`pass` only)

- `diagnostic_overlay.py::stop` — método de paragem não implementado
- `enterprise/agents/base.py::process` — stub de agente enterprise
- `enterprise/agents/base.py::think` — stub de agente enterprise
- `enterprise/agents/learning.py::_apply_transferred_knowledge` — stub
- `enterprise/vision/pipeline.py::load` — stub de pipeline vision
- `pylaai_real/dashboard_server.py::log_message` — logging handler vazio
- `training/auto_labeler.py::__init__` — inicialização vazia

### 2.3 Phase 10 modules — inicializados mas potencialmente não usados

O `wrapper.py` inicializa dezenas de módulos "Phase 10" com `try/except`, mas muitos nunca são de facto integrados no ciclo principal:

- `CentralCoordinator` — inicializado, não usado no loop
- `WorldModel` — inicializado, não usado no loop
- `OccupancyGrid` — inicializado, não usado no loop
- `PressureMap` — inicializado, não usado no loop
- `LobbyFSM` — inicializado no StateManager, mas `HAS_LOBBY_FSM` é False na prática
- `AsyncPipeline` — inicializado, não usado no loop
- `AdaptiveScreenshotCache` — inicializado, não usado no loop
- `BehavioralProfile` — inicializado, não usado no loop
- `InputOptimizer` — inicializado, não usado no loop
- `ReplayAnalyzer` — inicializado, não usado no loop
- `TacticalBridge` — inicializado, não usado no loop
- `CoverSystem` — inicializado, não usado no loop
- `UtilityAI`, `StickyTarget`, `IntentSystem`, `EnemyIntentionPredictor`, `MetaAwareness` — inicializados, não integrados

---

## 3. Utilidade Real do Bot

### 3.1 O que REALMENTE FUNCIONA

1. **Screenshot capture** — `ScreenshotTaker` (Win32) + fallback ADB
2. **State detection** — `UnifiedStateDetector` usa OpenCV com pixel matching + template matching (43KB de código real)
3. **State machine** — `StateManager.run()` cicla continuamente, chama handlers apropriados (lobby, loading, in_game, etc.)
4. **Lobby automation** — `LobbyAutomator` + `BrawlerQueue` clica no botão Play e seleciona brawlers
5. **Combat** — `PlayLogic.play_round()`:
   - Deteta jogador, inimigos, bushes, power cubes via YOLO
   - Calcula distâncias e escolhe alvo mais próximo
   - Envia swipes/taps via `emulator_controller`
   - Tem cooldowns, leading shot, kiting logic
6. **Safety system** — limites de trophies, APM, pausas obrigatórias
7. **Humanization** — delays, curvas Bezier, jitter de coordenadas
8. **RL Online** — `OnlineLearner` tem Q-table tabular real, guarda em `data/q_table.pkl`, usa epsilon-greedy
9. **ELO tracker** — ratings por brawler+map, guarda em JSON
10. **Dashboard** — servidor HTTP real com dados ao vivo do bot
11. **Replay Recorder** — grava screenshots + estado + ações
12. **A/B Testing** — compara variantes de estratégia

### 3.2 Funcionalidades FANTASMA (documentadas mas não implementadas)

| Funcionalidade | Documentada em | Estado Real | Impacto |
|----------------|----------------|-------------|---------|
| Wall / obstacle detection | `AGENTS.md` Phase 1 | **Não existe** — `movement.py` aceita `walls` como param mas ninguém deteta paredes | Médio — bot não evita paredes |
| OCR de nomes de mapas | `AGENTS.md` Known Limitations | **Não implementado** — `_current_map` é sempre None ou hardcoded | Baixo — mapas não afetam gameplay diretamente |
| Extração real de HP do jogador | `AGENTS.md` Known Limitations | **Não implementado** — `hp_estimate` é heuristic (1.0 default) | Alto — bot não sabe quando fugir baseado em HP real |
| YOLO inference paralela | `AGENTS.md` Phase 2 | **Não implementado** — inference é sequencial | Baixo — performance aceitável |
| Advanced Combat v2 Cover Engine | `AGENTS.md` | Parcial — código existe em `combat_advanced.py` mas não está totalmente ligado ao `play.py` | Médio |
| Real match rewards (RL) | `AGENTS.md` Known Limitations | **Heuristic rewards** — não extrai recompensas reais do jogo | Médio — RL aprende com heurísticas, não recompensas reais |
| Anti-ban advanced | `AGENTS.md` | `anti_ban_advanced.py` não é importado pelo wrapper | Baixo — anti-ban básico funciona |

### 3.3 Problemas de Integração

- **ScreenAutomation thread vs UnifiedStateDetector**: O `wrapper.py` explicitamente diz "ScreenAutomation thread NÃO é mais iniciada", mas ainda cria o objeto para "hints". Não há conflito real, mas é código morto.
- **EmulatorController como screenshot source**: Criado um `EmulatorWrapper` para dar `.take()` ao controller. Funciona, mas é uma abstração extra.
- **Match result tracking**: `MatchController` e `BrawlerStatsTracker` existem, mas a chamada de `record_match()` no ciclo principal não está claramente ligada ao `state_manager` — pode não estar a registar resultados automaticamente.

---

## 4. Dashboard — Verificação Zero Mock

### 4.1 Arquitetura

A dashboard usa `DashboardDataBridge` — buffer thread-safe atualizado pelo wrapper a cada ciclo do `_monitor_loop`. O wrapper chama `dashboard.update_from_wrapper(self)` periodicamente.

### 4.2 Origem dos Dados por Campo

| Campo | Origem | Real? |
|-------|--------|-------|
| `running` | `wrapper.running` | ✅ Sim |
| `current_state` | `state_manager.current_state` | ✅ Sim |
| `brawler` | `state_manager.current_brawler` ou `brawler_queue.get_current()` | ✅ Sim |
| `map_name` | `state_manager._current_map` | ⚠️ Sempre None (OCR não implementado) |
| `matches_total`, `wins`, `losses` | `observability.get_snapshot()` | ✅ Sim |
| `win_rate` | Calculado a partir de wins/matches | ✅ Sim |
| `cycle_time_ms`, `fps` | Calculado do observability | ✅ Sim |
| `epsilon`, `q_states` | `online_learner.get_stats()` | ✅ Sim (Q-table real) |
| `elo_combinations` | `online_learner.get_stats()` | ✅ Sim |
| `recent_events` | `observability.get_recent_events(20)` | ✅ Sim |
| `error_recovery_*` | `error_recovery.get_stats()` | ✅ Sim (se módulo carregar) |
| `state_recovery_*` | `state_recovery.get_recovery_status()` | ✅ Sim |
| `autocalibrator_*` | `auto_calibrator.get_all_cached_coords()` | ✅ Sim |
| `ocr_detector_*` | `ocr_detector.get_detection_stats()` | ✅ Sim (se módulo carregar) |
| `debug_visualizer_*` | `debug_visualizer.is_running` | ✅ Sim |
| `total_trophies` | `brawler_tracker.get_total_trophies()` | ✅ Sim (persiste em JSON) |
| `unlocked_brawlers` | `brawler_tracker.get_unlocked_count()` | ✅ Sim |
| `trophy_history` | `trophy_tracker.get_trophy_history()` | ✅ Sim (persiste em JSON) |
| `daily_evolution` | `trophy_tracker.get_daily_evolution()` | ✅ Sim |
| `weekly_progress` | `trophy_tracker.get_weekly_progress()` | ✅ Sim |
| `brawler_stats` | `brawler_tracker.get_all_stats()` | ✅ Sim (persiste em JSON) |
| `ai_pick_suggestion` | `match_analyzer.suggest_pick()` | ⚠️ Hardcoded rules, não aprendido |
| `win_prediction` | `match_analyzer.predict_win()` | ⚠️ Hardcoded heuristics |
| `coach_tips` | `match_analyzer.get_coach_tips()` | ⚠️ Hardcoded tips |
| `enemies_detected` | `play_logic._last_enemies` | ✅ Sim |
| `combat_mode` | `play_logic._last_combat_mode` / `advanced_combat.current_state` | ✅ Sim |
| `hp_estimate` | `advanced_combat.estimated_hp` | ⚠️ Heuristic (não extrai HP real) |
| `screenshot_b64` | Screenshot real do jogo (JPEG thumbnail) | ✅ Sim |
| `uptime_seconds` | `time.time() - session_start` | ✅ Sim |

### 4.3 Conclusão Dashboard

**Zero dados mock confirmado.** Nenhuma keyword `mock`, `demo`, `sample`, `fake`, ou `placeholder` encontrada no `dashboard_server.py`. Todos os dados vêm de instâncias reais do bot em execução.

**Ressalvas:**
- `MatchAnalyzer` usa dados hardcoded (BRAWLER_ROLES, MAP_ADVANTAGES, COUNTERS) — não é "mock", mas é estático e não evolui com o tempo.
- `hp_estimate` é heurístico (assume HP=1.0 quando não sabe), não extraído da UI.
- `map_name` é None na prática porque OCR de mapas não está implementado.

---

## 5. Melhorias Recomendadas (Priorizadas)

### 🔴 Crítico — Remover Código Morto

1. **Remover pasta `enterprise/` inteira** (74 itens) — nenhum é importado, ocupa espaço e confunde.
2. **Remover pasta `automation/` inteira** — não integrada.
3. **Remover `api/`** — API não ligada ao wrapper.
4. **Remover `training/` exceto pipeline ativo** — ou mover para branch separada.
5. **Remover `main.py`, `main_enhanced.py`, `lobby_runner.py`, `setup_wizard.py`** — entry points não usados.
6. **Remover stubs Phase 10 não usados** — `WorldModel`, `OccupancyGrid`, `PressureMap`, `LobbyFSM`, `AsyncPipeline`, etc., ou implementá-los de vez.

### 🟡 Alto — Funcionalidades Fantasma

7. **Implementar deteção de paredes** — usar diferença de cor/edge detection no screenshot para identificar obstáculos. Sem isto o bot atira contra paredes.
8. **Implementar extração de HP real** — OCR ou pixel matching nas barras de HP verdes/vermelhas no ecrã.
9. **Ligar `BrawlerStatsTracker.record_match()` ao ciclo de fim de partida** — garantir que resultados são realmente gravados.

### 🟢 Médio — Polimento

10. **Remover Phase 10 lazy imports do wrapper** se não forem usados — poluem o startup.
11. **Unificar entry points** — ter apenas `wrapper.py` como ponto de entrada oficial.
12. **Atualizar `AGENTS.md`** — remover funcionalidades que já estão implementadas e marcar claramente as fantasma.
13. **Cache de templates de estado** — o `UnifiedStateDetector` recarrega templates a cada ciclo? Verificar se há caching.

---

## 6. Métricas do Projeto

| Métrica | Valor |
|---------|-------|
| Total ficheiros Python | ~160 |
| Módulos nunca importados | 70+ |
| Funções com `pass` only | 7 |
| Ficheiros de log/test antigos | 30+ |
| Módulos realmente funcionais no loop principal | ~15 |
| Dados mock na dashboard | 0 |
| Funcionalidades fantasma | 6+ |

---

*Relatório gerado automaticamente por análise de código.*
