# PLANO DE MODERNIZAÇÃO E APRIMORAMENTO - SOBERANA OMEGA

**Versão:** 2.0 (Atualizado)
**Data:** 2026-05-15
**Status:** EM ANDAMENTO - Quick Wins 1-4 Implementados

---

## RESUMO EXECUTIVO

O projeto Soberana Omega Brawl Stars Bot encontra-se em **estado operacional** com módulos Phase 10 implementados e integrados. Após análise detalhada, muitos módulos "não integrados" na verdade JÁ estavam funcionando corretamente.

**Quick Wins Implementados (2026-05-15):**
1. ✅ BehavioralProfile save/load + shutdown hook
2. ✅ HP Extraction conectado ao Q-Learning
3. ✅ pyproject.toml criado
4. ✅ A* Pathfinding integrado para RETREAT

---

## ESTADO ATUAL DOS MÓDULOS

### ✅ JÁ INTEGRADOS E FUNCIONANDO

| Módulo | Status | Onde é Usado |
|--------|--------|--------------|
| UtilityAI | ✅ Integrado | `play.py:883` |
| CentralCoordinator | ✅ Integrado | `play.py:897-958` |
| IntentSystem | ✅ Integrado | `play.py:825-840` |
| StickyTarget | ✅ Integrado | `play.py:743-754, 917-928` |
| Q-Table Persistence | ✅ | Save on shutdown |
| ELO Persistence | ✅ | Save on shutdown |
| BehavioralProfile Persistence | ✅ | Save on shutdown (novo) |
| HP Extraction → RL | ✅ | Q-Learning usa HP real (novo) |
| A* Pathfinding | ✅ | RETREAT usa A* (novo) |

### ⚠️ INTEGRAÇÃO PARCIAL

| Módulo | Status | Problema |
|--------|--------|----------|
| WorldModel | ⚠️ Atualiza dados | Não influencia decisões |
| PressureMap | ⚠️ Atualiza dados | Não usado pelo UtilityAI |
| CoverSystem | ⚠️ Instanciado | Nunca chamado |
| EnemyIntention | ⚠️ Instanciado | Nunca chamado |
| MetaAwareness | ⚠️ Instanciado | Nunca chamado |

---

## CRONOGRAMA GERAL

| Fase | Período | Foco Principal |
|------|---------|----------------|
| 1 | Semana 1 | Investigação e Estabilização |
| 2 | Semanas 2-3 | Integração da Arquitetura de IA Phase 10 |
| 3 | Semana 4 | Visão Computacional e Navegação Real |
| 4 | Semana 5 | Ciclo de Feedback e Aprendizado |
| 5 | Semana 6 | Automação de Testes e CI/CD |

---

## FASE 1: INVESTIGAÇÃO E ESTABILIZAÇÃO

**Período:** Semana 1
**Responsável:** Equipo Completo
**Objetivo:** Estabelecer base sólida para desenvolvimento

### 1.1 Reestruturação do Repositório `pylaai_real/`

**Descrição:**
Migrar a estrutura do diretório `pylaai_real/` para formato versionado e escalável, seguindo PEP 517/518/621.

**Atividades:**
- Criar `pyproject.toml` com dependências formally definidas (ultralytics, opencv-python, numpy, pillow, pytesseract, etc.)
- Reorganizar `pylaai_real/` em subpacotes funcionais se necessário
- Criar arquivo `pylaai_real/__version__.py` para controle de versão semântico
- Adicionar `CHANGELOG.md` para rastreamento de alterações
- Verificar compatibilidade de todos os imports relativos/absolutos

**Entregáveis:**
- `pyproject.toml` validado com `pip install -e .`
- `CHANGELOG.md` criado com estrutura de release
- Imports absolutos funcionando em todos os módulos

**Critérios de Sucesso:**
- `python -c "import pylaai_real"` executa sem erros
- `pip install -e .` completa sem warnings
- Git tags funcionam para versionamento

**Padrões de Qualidade:**
- Seguir Semantic Versioning 2.0.0
- Usar importações absolutas (ex: `from brawl_bot.pylaai_real import ...`)
- Documentar todas as dependências com versão mínima

---

### 1.2 Testes End-to-End (5 Partidas Integrais)

**Descrição:**
Executar validação completa do bot através de 5 partidas reais para identificar instabilidades.

**Atividades:**
- Desenvolver `tests/e2e/test_full_match_flow.py`:
  - Conectar ao emulador
  - Entrar no lobby automaticamente
  - Selecionar brawler correto
  - Aguardar loading
  - Jogar partida completa
  - Processar tela de fim (vitória/derrota)
  - Retornar ao lobby
- Executar 5 partidas com logging detalhado
- Documentar todas as falhas e instabilidades
- Criar matriz de risco dos problemas encontrados

**Entregáveis:**
- Script de teste E2E funcional
- Logs de 5 partidas completas
- `TEST_REPORT.md` com análise de instabilidades
- Lista priorizada de bugs encontrados

**Critérios de Sucesso:**
- 5/5 partidas completadas (mesmo com glitches)
- Taxa de sucesso lobby→jogo > 80%
- Taxa de sucesso seleção de brawler > 70%
- 100% dos erros logados com screenshot

**Padrões de Qualidade:**
- Usar pytest fixtures para setup/teardown
- Capturar screenshot a cada falha
- Timestamps em todos os logs

---

### 1.3 Atualização do README.md

**Descrição:**
Documentar completamente o estado atual do projeto.

**Atividades:**
- Reescrever README.md com seções:
  1. Badges (CI, version, Python)
  2. Descrição do projeto
  3. Arquitetura do sistema (com diagrama ASCII)
  4. Requisitos de sistema
  5. Instalação passo-a-passo
  6. Como executar
  7. Configuração
  8. API Reference (links)
  9. Testes (como executar)
  10. Limitaciones conhecidas
  11. Roadmap de desenvolvimento
  12. Contribuição
  13. Licença
- Adicionar diagramas de arquitetura
- Criar seção de troubleshooting

**Entregáveis:**
- README.md com mínimo 500 linhas
- Todos os links funcionais
- Badges de status

**Critérios de Sucesso:**
- Documentação cobrindo 100% das features
- Instruções de instalação testadas por terceiros
- Screenshots da UI funcionais

---

### 1.4 Correção de Bugs Críticos

**Descrição:**
Identificar e corrigir todos os bugs que impedem funcionamento básico.

**Atividades:**
- Executar suite completa de testes: `pytest tests/ -v`
- Classificar bugs por severidade:
  - **Crítico:** Impede funcionamento básico (crash, deadlock)
  - **Major:** Afeta funcionalidade principal (detecção falhando)
  - **Minor:** Inconveniência (UI bug, warning)
- Corrigir bugs críticos 먼저
- Executar testes de regressão após cada correção
- Documentar bugs no CHANGELOG

**Entregáveis:**
- Lista de bugs classificados por severidade
- Correções aplicadas
- Testes de regressão passando

**Critérios de Sucesso:**
- Zero bugs críticos
- Suite de testes passando > 90%
- Zero regressões em funcionalidades existentes

---

## FASE 2: INTEGRAÇÃO DA ARQUITETURA DE IA PHASE 10

**Período:** Semanas 2-3
**Responsável:** AI/ML Team
**Objetivo:** Integrar módulos de IA avançados ao loop principal

### 2.1 Integração do UtilityAI ao Loop Principal

**Descrição:**
Conectar o sistema UtilityAI ao wrapper.py para decisões scored baseadas em contexto.

**Atividades:**
- Integrar `decision/utility_ai.py` ao `wrapper.py`:
  ```python
  from decision.utility_ai import UtilityAI, Action, ActionScore

  utility_ai = UtilityAI()
  # No loop de decisão:
  scores = utility_ai.evaluate_actions(world_state, available_actions)
  best_action = utility_ai.select_best(scores)
  ```
- Conectar com WorldModel para contexto de jogo
- Implementar pesos configuráveis por perfil de jogo
- Criar logging de scores para debugging

**Entregáveis:**
- UtilityAI processando decisões a cada frame
- Logs de ActionScore para cada decisão
- Comportamento adaptativo observável

**Critérios de Sucesso:**
- UtilityAI avaliando todas as ações disponíveis
- Logs mostrando score breakdown
- Decisões sendo tomadas baseadas em contexto

**Padrões de Implementação:**
```python
@dataclass
class ActionScore:
    action: Action
    score: float
    reasoning: str
    target: Optional[Tuple[int, int]]
```

---

### 2.2 Integração do CentralCoordinator

**Descrição:**
Implementar resolução de conflitos entre subsistemas de IA.

**Atividades:**
- Integrar `core/central_coordinator.py`:
  - Conectar UtilityAI, PressureMap, IntentSystem, StickyTarget
  - Implementar Priority enum (CRITICAL > HIGH > MEDIUM > LOW)
  - Criar Recommendation dataclass dos subsistemas
  - Implementar resolução por confiança ponderada
- Conectar saída ao PlayLogic e Movement
- Implementar CONSISTENCY_BIAS para manter decisões

**Entregáveis:**
- CentralCoordinator receber recomendações de todos subsistemas
- Logs de resolução de conflitos
- Decisão unificada sendo executada

**Critérios de Sucesso:**
- Logs mostrando recomendações de múltiplas fontes
- Conflitos resolvidos com base em prioridade
- Decisão final sem contradições

---

### 2.3 Integração do IntentSystem

**Descrição:**
Implementar persistência de objetivos estratégicos ao longo da partida.

**Atividades:**
- Integrar `decision/intent_system.py`:
  - Conectar ao estado do jogo (fase da partida)
  - Implementar MIN_INTENT_DURATION_MS = 2000ms
  - Integrar com modo de jogo (MODE_DEFAULTS)
  - Implementar HYSTERESIS_MARGIN para mudanças
- Criar transições de intent baseadas em eventos:
  - FARM → AGGRESSIVE (3+ cubes)
  - AGGRESSIVE → SURVIVE (HP < 30%)
  - SURVIVE → FARM (HP full, inimigos distantes)

**Entregáveis:**
- Intent persistindo por mínimo 2 segundos
- Logs mostrando transições de intent
- Comportamento diferente por modo de jogo

**Critérios de Sucesso:**
- Intent não muda antes de 2000ms sem justificativa forte
- Transições documentadas nos logs
- Perfis de jogo visíveis no comportamento

---

### 2.4 Integração do StickyTarget

**Descrição:**
Implementar comprometimento com alvos para evitar thrashing.

**Atividades:**
- Integrar `decision/sticky_target.py`:
  - Conectar ao sistema de targeting do PlayLogic
  - Implementar min_commitment_ms = 800ms
  - Implementar hysteresis_margin = 0.3
  - Conectar com tracker de inimigos
- Adicionar métricas de effectiveness:
  - focus_time (tempo no target)
  - shots_at_target
  - hit_rate

**Entregáveis:**
- Target mantido por mínimo 800ms
- Redução mensurável de target switching
- Logs mostrando reason para eventual troca

**Critérios de Sucesso:**
- target_thrashing reduzido em 50% vs baseline
- Logs com CommitmentReason documentado
- Focus bonus observável no gameplay

---

### 2.5 Testes de Integração (10 Partidas)

**Descrição:**
Validar comportamento coordenado com todos os módulos de IA integrados.

**Atividades:**
- Executar 10 partidas reais comlogging completo
- Monitorar:
  - Comunicação entre UtilityAI e CentralCoordinator
  - Persistência de IntentSystem
  - Commitment do StickyTarget
  - Resolução de conflitos
- Medir win rate e сравнить com baseline
- Documentar métricas de integração

**Entregáveis:**
- Logs de 10 partidas completas
- Métricas de integração (ação por minuto, trocas de target, etc.)
- Win rate comparativo

**Critérios de Sucesso:**
- 10/10 partidas completadas
- Win rate mantido ou melhorado
- Zero deadlocks ou race conditions
- Logs mostrando fluxo correto de dados

---

## FASE 3: VISÃO COMPUTACIONAL E NAVEGAÇÃO REAL

**Período:** Semana 4
**Responsável:** Vision Team
**Objetivo:** Implementar detecção e navegação com alta precisão

### 3.1 Sistema de Detecção de Paredes (95% Acerto)

**Descrição:**
Implementar detecção de obstáculos em tempo real com taxa de acerto mínima de 95%.

**Atividades:**
- Avaliar abordagens:
  - **OpenCV color/edge detection** (fallback atual)
  - **Template matching** (mais rápido para paredes fixas)
  - **YOLO customizado** (melhor mas requer training)
- Implementar `vision/game_feature_extractor.py:detect_walls()`
- Validar em 5 mapas diferentes
- Otimizar para < 100ms por frame

**Entregáveis:**
- Sistema de detecção de paredes funcionando
- Precisão > 95% em cenários de teste
- Latência < 100ms por frame

**Critérios de Sucesso:**
- Testes em 5 mapas com > 95% acerto
- FPS não reduzido em mais de 2 com detecção ativa
- Detecção funcionando para paredes fixas e dinâmicas

---

### 3.2 Extração de HP em Tempo Real (<100ms)

**Descrição:**
Implementar leitura de pontos de vida com atualização inferior a 100ms.

**Atividades:**
- Implementar `game_feature_extractor.py:extract_player_hp()`:
  - Localizar HP bar acima do player
  - Analisar pixels verdes vs vermelhos
  - Calcular ratio: HP = green / (green + red)
- Explorar alternativas:
  - **Memória direta** (mais rápido mas complexo)
  - **OCR** (mais genérico mas mais lento)
- Implementar cache de HP (150ms TTL)
- Validar precisão vs valores reais

**Entregáveis:**
- Sistema de extração de HP funcionando
- Atualização < 100ms
- Precisão > 90%

**Critérios de Sucesso:**
- HP atualizado em < 100ms
- Precisão > 90% vs valor real do jogo
- Funcionando para player e inimigos

---

### 3.3 Validação do OccupancyGrid

**Descrição:**
Validar que o OccupancyGrid representa corretamente o ambiente detectado.

**Atividades:**
- Integrar detecções de paredes ao OccupancyGrid
- Testar alinhamento 100% entre detecção e grid:
  - Verificar que células marcadas como WALL correspondem a paredes
  - Testar em mapas com obstáculos complexos
- Implementar validação por raycasting:
  - A cada frame, verificar LOS do player aos inimigos
  - Comparar com detecção real
- Medir taxa de alinhamento

**Entregáveis:**
- OccupancyGrid 100% alinhado com obstáculos
- Testes de validação passando
- Relatório de precisão

**Critérios de Sucesso:**
- Grid cells = WALL para 100% das paredes detectadas
- Zero falsos positivos (espaço walkable marcado como wall)
- Atualização dinâmica durante partidas

---

### 3.4 Sistema de Navegação A* (100% Sucesso)

**Descrição:**
Validar que o bot calcula e executa rotas válidas sem colisões.

**Atividades:**
- Implementar e validar A* pathfinding em `core/occupancy_grid.py`:
  - Verificar que find_path() retorna rota válida
  - Testar em cenários com múltiplos obstáculos
  - Implementar validação de rota antes de executar
- Testar 100 cenários de pathfinding
- Medir:
  - Taxa de sucesso (rota encontrada)
  - Tempo de cálculo
  - Ocorrência de colisões durante execução

**Entregáveis:**
- A* pathfinding funcionando
- 100% das rotas calculadas são válidas
- Zero colisões durante execução

**Critérios de Sucesso:**
- Rota encontrada em < 50ms para 95% dos casos
- Zero colisões em 100 testes de navegação
- Fallback gracioso quando rota não existe

---

## FASE 4: CICLO DE FEEDBACK E APRENDIZADO

**Período:** Semana 5
**Responsável:** RL/ML Team
**Objetivo:** Implementar ciclo de feedback validado empiricamente

### 4.1 Sistema de Validação de Ações

**Descrição:**
Implementar verificação automatizada se comandos foram concluídos com sucesso.

**Atividades:**
- Desenvolver `ActionValidator`:
  - Detectar se ataque hitou (damage dealt > 0)
  - Verificar se movimento foi executado (posição mudou)
  - Validar uso de super (cooldown resetou)
  - Checar coleta de cube (HP/cubes aumentaram)
- Integrar com sistema de logging
- Criar métricas de sucesso por tipo de ação

**Entregáveis:**
- ActionValidator validando 100% das ações
- Logs estruturados com sucesso/falha
- Métricas por tipo de ação

**Critérios de Sucesso:**
- 100% das ações validadas
- Taxa de sucesso de ataques logada corretamente
- Movimento validado por posição final

---

### 4.2 Sistema de Logging de Partidas

**Descrição:**
Implementar logging estruturado de todas as partidas para análise.

**Atividades:**
- Desenvolver `MatchLogger`:
  - Estado inicial (brawler, mapa, modo)
  - Ações executadas com timestamps
  - Resultados de cada ação (sucesso/falha)
  - HP do player ao longo do tempo
  - Inimigos eliminados / morte
  - Resultado final (vitória/derrota)
- Formato JSON para parse fácil:
  ```json
  {
    "match_id": "uuid",
    "timestamp": "ISO8601",
    "brawler": "Shelly",
    "map": "Hard Rock Mine",
    "mode": "showdown",
    "result": "victory",
    "actions": [...],
    "events": [...]
  }
  ```
- Implementar rotação de logs (max 100MB por arquivo)

**Entregáveis:**
- Logs de 100% das partidas em formato JSON
- Estrutura parseável
- Sem perda de dados

**Critérios de Sucesso:**
- Todos os campos preenchidos corretamente
- JSON válido e parseável
- Logs ocupando < 100MB por partida

---

### 4.3 Sistema de Reward Real

**Descrição:**
Substituir reward heurístico por reward baseado em resultados concretos.

**Atividades:**
- Implementar reward system baseado em eventos:
  - **Damage dealt:** +0.01 por dano unitário
  - **Kill:** +10.0
  - **Death:** -5.0
  - **Cube collected:** +1.0
  - **Victory:** +50.0
  - **Defeat:** -20.0
  - **Survived:** +0.1 por segundo vivo
- Integrar com `pylaai_real/rl_engine.py` existente
- Implementar reward shaping para convergência
- Validar correlação com win rate

**Entregáveis:**
- Sistema de reward implementado
- Q-table atualizando após cada partida
- Reward correlacionado com resultado

**Critérios de Sucesso:**
- Reward positivo para vitórias
- Reward negativo para derrotas
- Q-table convergindo após 20 partidas

---

### 4.4 Testes do Ciclo Completo (20 Partidas)

**Descrição:**
Validar ciclo completo: coleta → reward → ajuste → melhoria.

**Atividades:**
- Executar 20 partidas consecutivas
- Monitorar:
  - Coleta de dados (100% das ações logadas)
  - Processamento de reward (calculado corretamente)
  - Atualização de Q-table ( pesos mudando)
  - Ajuste de estratégia (comportamento melhorando)
- Comparar performance: partidas 1-10 vs 11-20

**Entregáveis:**
- Logs de 20 partidas completas
- Análise de melhoria de win rate
- Q-table salvo após cada partida

**Critérios de Sucesso:**
- Sistema coletando dados consistentemente
- Q-table atualizando após cada partida
- Win rate melhorando (1-10 avg vs 11-20 avg)

---

## FASE 5: AUTOMAÇÃO DE TESTES E CI/CD

**Período:** Semana 6
**Responsável:** DevOps
**Objetivo:** Garantir qualidade contínua com automação

### 5.1 Testes End-to-End Completos

**Descrição:**
Desenvolver suite E2E cobrindo todo fluxo de uso do bot.

**Atividades:**
- Desenvolver `tests/e2e/test_lobby_to_end.py`:
  - Teste 1: Lobby → Seleção de Brawler
  - Teste 2: Seleção → Loading da partida
  - Teste 3: Loading → Gameplay
  - Teste 4: Gameplay → Tela de fim
  - Teste 5: Fim → Retorno ao Lobby
- Implementar `tests/e2e/test_state_transitions.py`:
  - Testar cada estado: lobby, select, loading, in_game, victory, defeat, error
  - Validar transições corretas
- Criar fixtures pytest para emulador
- Gerar relatório de cobertura

**Entregáveis:**
- Suite E2E cobrindo 100% dos fluxos
- Testes executando em < 30 minutos
- Relatório de cobertura de código

**Critérios de Sucesso:**
- 100% dos fluxos de uso cobertos
- Suite E2E passando > 95%
- Cobertura de código > 80%

---

### 5.2 Testes de Integração Phase 10

**Descrição:**
Validar comunicação entre módulos de IA implementados.

**Atividades:**
- Desenvolver `tests/test_ai_integration.py`:
  - UtilityAI → CentralCoordinator communication
  - IntentSystem persistence over time
  - StickyTarget commitment validation
  - WorldModel updates from detections
- Implementar `tests/test_pathfinding_integration.py`:
  - OccupancyGrid updates from wall detection
  - A* pathfinding with dynamic obstacles
- Criar testes de race conditions

**Entregáveis:**
- Testes de integração para cada módulo
- Validação de comunicação entre componentes
- Relatório de race conditions

**Critérios de Sucesso:**
- Todos os módulos comunicando corretamente
- Zero deadlocks ou race conditions
- Logs mostrando fluxo correto

---

### 5.3 Testes de Performance

**Descrição:**
Implementar testes que medem métricas críticas de performance.

**Atividades:**
- Implementar `tests/performance/test_fps.py`:
  - FPS médio durante gameplay
  - FPS mínimo (nadir)
  - FPS por fase (lobby vs in_game)
- Implementar `tests/performance/test_latency.py`:
  - Latência: evento detectado → ação executada
  - Latência máxima e média
  - Percentis (p50, p95, p99)
- Implementar `tests/performance/test_resources.py`:
  - Uso de memória ao longo do tempo
  - Uso de CPU por componente
  -Throughput de detecções por segundo

**Entregáveis:**
- Suite de performance funcionando
- Baseline de performance estabelecido
- Relatório comparativo

**Critérios de Sucesso:**
- FPS médio > 15 durante gameplay
- Latência p95 < 500ms
- Memória < 2GB, CPU < 80%

---

### 5.4 Pipeline de CI/CD

**Descrição:**
Configurar integração contínua para execução automática de testes.

**Atividades:**
- Configurar GitHub Actions (`.github/workflows/ci.yml`):
  ```yaml
  on: [push, pull_request]
  jobs:
    lint:
      - runs: ruff check .
      - runs: mypy src/
    test:
      - runs: pytest tests/ -v
    performance:
      - runs: pytest tests/performance/ --benchmark
    build:
      - runs: pip install -e .
      - runs: pytest tests/ --integration
  ```
- Configurar notifications (Discord webhook)
- Implementar relatórios de performance (JSON → HTML)
- Adicionar badges ao README

**Entregáveis:**
- Pipeline CI/CD configurado
- Relatórios de performance gerados
- Badges no README

**Critérios de Sucesso:**
- Pipeline executando em cada push
- Relatórios disponíveis para cada run
- Notificações enviadas em falhas

---

## MATRIZ DE DEPENDÊNCIAS

```
FASE 1 (Semana 1)
├── 1.1 pyproject.toml
│   └── Requer: None
├── 1.2 Testes E2E
│   └── Requer: 1.1 (build funcionando)
├── 1.3 README.md
│   └── Requer: 1.2 (saber estado real)
└── 1.4 Correção de Bugs
    └── Requer: 1.2 (bugs identificados nos testes)

FASE 2 (Semanas 2-3)
├── 2.1 UtilityAI
│   └── Requer: 1.1, 1.4
├── 2.2 CentralCoordinator
│   └── Requer: 2.1
├── 2.3 IntentSystem
│   └── Requer: 2.1
├── 2.4 StickyTarget
│   └── Requer: 2.1
└── 2.5 Testes (10 partidas)
    └── Requer: 2.2, 2.3, 2.4

FASE 3 (Semana 4)
├── 3.1 Detecção de Paredes
│   └── Requer: 1.1
├── 3.2 Extração de HP
│   └── Requer: 3.1
├── 3.3 OccupancyGrid
│   └── Requer: 3.1
└── 3.4 A* Pathfinding
    └── Requer: 3.3

FASE 4 (Semana 5)
├── 4.1 ActionValidator
│   └── Requer: 3.1
├── 4.2 MatchLogger
│   └── Requer: 4.1
├── 4.3 Reward System
│   └── Requer: 4.2, 2.1
└── 4.4 Testes (20 partidas)
    └── Requer: 4.3

FASE 5 (Semana 6)
├── 5.1 Testes E2E
│   └── Requer: 4.1
├── 5.2 Testes Integração AI
│   └── Requer: 2.5
├── 5.3 Testes Performance
│   └── Requer: 3.4
└── 5.4 CI/CD
    └── Requer: 5.1, 5.2, 5.3
```

---

## CRITÉRIOS DE QUALIDADE GERAIS

| Métrica | Target | Medição |
|---------|--------|---------|
| Cobertura de Código | > 80% | pytest-cov |
| Testes Passando | > 95% | pytest |
| FPS em Gameplay | > 15 | benchmarks |
| Latência p95 | < 500ms | benchmarks |
| Memória | < 2GB | benchmarks |
| CPU | < 80% | benchmarks |
| Documentação | 100% módulos | code review |

---

## RISCOS E MITIGAÇÕES

| Risco | Probabilidade | Impacto | Mitigação |
|-------|---------------|---------|-----------|
| Detecção de paredes < 95% | Alta | Médio | Ter fallback com templates |
| Instabilidade emuladores | Alta | Alto | Retry logic + screenshot validation |
| Integração UtilityAI complexa | Média | Alto | Implementar incrementalmente |
| Performance degradation | Média | Médio | Benchmarks contínuos |
| CI/CD infrastructure | Baixa | Médio | Usar GitHub Actions (free tier) |

---

## PRÓXIMOS PASSOS

1. **Aprovar plano** - Revisar e validar com stakeholders
2. **Semana 1** - Iniciar Fase 1 (versionamento + testes E2E)
3. **Checkpoint** - Revisão após Semana 1 com resultados
4. **Iteração** - Ajustar plano baseado em aprendizados

---

*Documento gerado automaticamente para o projeto Soberana Omega Brawl Stars Bot*
*Versão para planejamento e execução de modernização*
