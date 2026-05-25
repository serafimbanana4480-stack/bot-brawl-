# PLANO DE REFATORAÇÃO E REDUÇÃO DE COMPLEXIDADE

**Data:** 2026-05-17
**Objetivo:** Reduzir código de ~419KB para <200KB (redução de 52%)

---

## 📊 ESTADO ATUAL

| Ficheiro | Tamanho | Linhas Est. | Problema |
|----------|---------|-------------|----------|
| `dashboard_server.py` | 153 KB | ~4500 | HTML inline + lógica混杂 |
| `wrapper.py` | 104 KB | ~2200 | God Object - faz tudo |
| `play.py` | 89 KB | ~1800 | Lógica de combate混杂 |
| `state_manager.py` | 73 KB | ~1500 | Handlers de estado mezclados |
| **TOTAL** | **419 KB** | **~10000** | |

---

## 🎯 OBJETIVOS DE REDUÇÃO

| Ficheiro | Tamanho Atual | Tamanho Alvo | Redução |
|----------|---------------|--------------|---------|
| `dashboard_server.py` | 153 KB | 60 KB | 61% |
| `wrapper.py` | 104 KB | 50 KB | 52% |
| `play.py` | 89 KB | 40 KB | 55% |
| `state_manager.py` | 73 KB | 35 KB | 52% |
| **TOTAL** | **419 KB** | **<185 KB** | **56%** |

---

## 🔴 PRIORIDADE 1: Módulos Decorativos (✅ JÁ REMOVIDOS)

**Sete módulos foram removidos:**
- `resilience_system.py` (~400 linhas)
- `perception_system.py` (~500 linhas)
- `decision_system.py` (~550 linhas)
- `behavioral_system.py` (~450 linhas)
- `active_learning_system.py` (~700 linhas)
- `intelligent_alerts.py` (~400 linhas)
- `professional_bot_core.py` (~350 linhas)

**Total removido:** ~3,350 linhas de código decorativo

---

## 🟡 PRIORIDADE 2: Refatoração dashboard_server.py

### Problema
- 153 KB com HTML inline (~1600 linhas de HTML)
- 11 classes mezcladas (BotLiveData, DashboardDataBridge, ReplayRecorder, etc.)
- Lógica de servidor HTTP mezclada com lógica de negócio

### Solução
1. **Extrair HTML** para ficheiro separado `dashboard_html.py`
2. **Separar classes** em módulos dedicados:
   - `dashboard_data.py` - BotLiveData, DashboardDataBridge
   - `dashboard_replay.py` - ReplayRecorder, ReplayFrame
   - `dashboard_abtest.py` - ABTestManager, ABTestVariant
   - `dashboard_stats.py` - BrawlerStatsTracker, MatchAnalyzer, TrophyTracker
   - `dashboard_server.py` - DashboardHandler, DashboardServer (só orquestração)
3. **Mover HTML** para `dashboard_web/` como ficheiro estático

### Resultado Esperado
- `dashboard_server.py`: 153 KB → 40 KB (redução de 74%)
- HTML externo: ~110 KB em ficheiro separado
- Classes separadas: melhor testabilidade

---

## 🔴 PRIORIDADE 3: Refatoração wrapper.py

### Problema
- 104 KB com ~2200 linhas
- Inicialização de 20+ subsistemas mezclados
- Loop principal mezclado com lógica de monitoring

### Solução
1. **Extrair inicializadores** para `wrapper_init.py`:
   - `_init_safety()`
   - `_init_humanization()`
   - `_init_emulator()`
   - `_init_detectors()`
   - `_init_play_logic()`
   - `_init_rl_engine()`
2. **Extrair loop de monitoring** para `wrapper_monitor.py`
3. **Manter** apenas `PylaAIEnhanced` orchestrating

### Resultado Esperado
- `wrapper.py`: 104 KB → 50 KB (redução de 52%)
- Ficheiros separados: melhor manutenção

---

## 🔴 PRIORIDADE 4: Refatoração play.py

### Problema
- 89 KB com ~1800 linhas
- Lógica de combate mezclada com pathfinding
- Utility AI mezclada com execução

### Solução
1. **Manter** `combat_advanced.py` como dependência principal
2. **Extrair** para `play_pathfinding.py`:
   - `_find_path_to_enemy()`
   - `_kiting_logic()`
   - `_cover_seeking()`
3. **Extrair** para `play_targeting.py`:
   - `_select_target()`
   - `_leading_shot_calculations()`
4. **Simplificar** `play.py` para só orquestração

### Resultado Esperado
- `play.py`: 89 KB → 40 KB (redução de 55%)
- Módulos separados: melhor testabilidade

---

## 🟢 PRIORIDADE 5: Refatoração state_manager.py

### Problema
- 73 KB com ~1500 linhas
- Handlers de estado (lobby, in_game, etc.) mezclados
- Lógica de deteção mezclada com transições

### Solução
1. **Extrair handlers** para `state_handlers.py`:
   - `_handle_lobby()`
   - `_handle_in_game()`
   - `_handle_tutorial()`
   - `_handle_news()`
   - `_safe_back_to_lobby()`
2. **Extrair** lógica de deteção para `state_detection.py`
3. **Simplificar** `state_manager.py` para só state machine

### Resultado Esperado
- `state_manager.py`: 73 KB → 35 KB (redução de 52%)
- Handlers separados: melhor manutenção

---

## 📋 PLANO DE EXECUÇÃO

### Fase 1: Redução Imediata (Esta Sessão)
- [x] Remover 7 módulos decorativos
- [x] Documentar plano de refatoração

### Fase 2: Refatoração wrapper.py (Próxima Sessão)
- [ ] Criar `wrapper_init.py` com inicializadores
- [ ] Criar `wrapper_monitor.py` com loop de monitoring
- [ ] Simplificar `wrapper.py` para orquestração

### Fase 3: Refatoração play.py
- [ ] Criar `play_pathfinding.py`
- [ ] Criar `play_targeting.py`
- [ ] Simplificar `play.py`

### Fase 4: Refatoração state_manager.py
- [ ] Criar `state_handlers.py`
- [ ] Simplificar `state_manager.py`

### Fase 5: Refatoração dashboard_server.py
- [ ] Extrair HTML para `dashboard_web/index.html`
- [ ] Separar classes em módulos
- [ ] Simplificar `dashboard_server.py`

---

## ✅ RESULTADO ESPERADO

| Métrica | Antes | Depois | Melhoria |
|---------|-------|--------|----------|
| Total KB | 419 KB | <185 KB | -56% |
| Total Linhas | ~10000 | <4500 | -55% |
| Módulos Decorativos | 8 | 0 | -100% |
| God Objects | 4 | 0 | -100% |

---

## 🎯 CRITÉRIOS DE SUCESSO

1. ✅ Redução de 50%+ no tamanho total dos 4 ficheiros
2. ✅ Cada módulo extraído tem testes unitários
3. ✅ Imports mantidos - nenhuma funcionalidade quebrada
4. ✅ Logs e erros preservados
5. ✅ Performance não degradada