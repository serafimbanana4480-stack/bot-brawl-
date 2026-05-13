# Plano de Melhorias - Dashboard Completa Soberana Omega

## Resumo
Transformar a dashboard atual (10 tabs, read-only + start/stop) num **Centro de Controle Total** onde se pode executar, configurar, monitorizar e analisar o bot sem tocar em codigo. Todas as melhorias sao feitas no ficheiro `pylaai_real/dashboard_server.py` e `wrapper.py`.

---

## Phase 1: Controle Total do Bot via Dashboard
**Objetivo:** Permitir pausar, configurar e controlar todos os sistemas do bot em tempo real.

### 1.1 Novos Endpoints REST (dashboard_server.py)
- `POST /api/bot/pause` -> chama wrapper.pause()
- `POST /api/bot/resume` -> chama wrapper.resume()
- `POST /api/bot/queue/update` -> JSON com nova fila de brawlers
- `POST /api/bot/queue/set-brawler` -> seleciona brawler manual
- `POST /api/bot/queue/add` -> adiciona brawler a fila
- `POST /api/bot/queue/remove` -> remove brawler da fila
- `POST /api/bot/config/update` -> atualiza central_config em tempo real
- `POST /api/bot/action/{action}` -> acoes manuais:
  - `force_goto_lobby`, `force_click_play`, `force_attack`, `force_super`, `force_collect_cube`
- `POST /api/system/toggle` -> liga/desliga sistemas:
  - `rl_engine`, `humanization`, `anti_ban`, `error_recovery`, `auto_tuner`, `recording`

### 1.2 Wrapper - Metodos Faltantes (wrapper.py)
- `pause()` -> pausa state_manager.run() sem matar threads
- `resume()` -> retoma
- `update_config(key, value)` -> atualiza config.json + central_config
- `set_brawler(name)` -> forca selecao de brawler
- `update_queue(new_queue)` -> substitui fila completa
- `toggle_system(system_name, enabled)` -> liga/desliga componentes em runtime
- `execute_action(action_name, **kwargs)` -> acoes manuais

### 1.3 UI - Painel de Controle (dashboard_server.py HTML)
- Novo card na tab "Tempo Real":
  - Botao **Pause / Resume** (toggle)
  - Dropdown **Selecionar Brawler**
  - Slider **Aggressiveness** (0-100%)
  - Slider **Shot Cooldown** (200-600ms)
  - Toggle switches: RL Engine, Humanizacao, Anti-Ban, Error Recovery, Gravacao
- Secao **Fila de Brawlers** (reordenavel, add/remove)

---

## Phase 2: Logs e Debug em Tempo Real
**Objetivo:** Ver o que o bot esta a fazer em tempo real, como um terminal.

### 2.1 Backend - Log Buffer (novo ficheiro: core/log_buffer.py)
- `LogBuffer` — buffer circular thread-safe (ultimas 500 linhas)
- Handlers custom para logging que alimentam o buffer
- Filtros por nivel (DEBUG/INFO/WARNING/ERROR) e componente

### 2.2 Endpoints REST
- `GET /api/logs?level=INFO&component=play&limit=100` -> ultimas linhas
- `GET /api/logs/stream` -> Server-Sent Events para logs em tempo real

### 2.3 UI - Nova Tab "Logs"
- Terminal-style log viewer com:
  - Filtro por nivel (botoes DEBUG/INFO/WARN/ERROR)
  - Filtro por componente (dropdown)
  - Auto-scroll toggle
  - Search/filter por texto
  - Color coding por nivel
  - Botao "Download Logs"

### 2.4 Health Monitor Visivel
- Novo card na tab "Tempo Real":
  - Status de cada componente (YOLO, OCR, ADB, RL, etc.) com indicador verde/amarelo/vermelho
  - Tempo desde ultima acao (deadlock detection)
  - Memoria/CPU do processo

---

## Phase 3: Notificacoes e Alertas
**Objetivo:** O bot avisa quando precisa de atencao, mesmo quando nao se esta a olhar para a dashboard.

### 3.1 Sistema de Notificacoes (novo ficheiro: core/notifications.py)
- `NotificationManager` com suporte a multiplos canais:
  - Browser notifications (via JS push no frontend)
  - Webhooks HTTP (Discord/Slack/custom)
  - Desktop notifications (via plyer/win10toast)
- Triggers configuraveis:
  - Bot crashou / parou
  - Derrotas seguidas >= N
  - Tempo em estado "unknown" >= N segundos
  - Trofeus atingiram limite de seguranca
  - Match terminou (win/loss)
  - Erro nao recuperado

### 3.2 Endpoints REST
- `GET /api/notifications/config` -> configuracao atual
- `POST /api/notifications/config` -> atualiza configuracao
- `POST /api/notifications/test` -> testa notificacao

### 3.3 UI - Nova Tab "Notificacoes"
- Configuracao de webhooks (URL, headers)
- Toggle Browser Notifications
- Lista de triggers configuraveis com thresholds
- Historico de notificacoes enviadas
- Botao "Testar Notificacao"

---

## Phase 4: Editor de Configuracao
**Objetivo:** Editar todas as configuracoes do bot sem abrir ficheiros.

### 4.1 Backend
- `POST /api/config` -> retorna config.json completo
- `POST /api/config/update` -> atualiza e grava no disco
- Validacao de tipos e ranges
- Backup automatico da config anterior

### 4.2 UI - Nova Tab "Configuracao"
- Editor JSON com syntax highlighting (ou formulario estruturado)
- Secoes:
  - **Emulador**: device_id, adb_path, window matching
  - **Modelos**: caminhos dos modelos YOLO, conf thresholds
  - **Safety**: max_trophies, max_apm, session_duration, break_interval
  - **Humanizacao**: delay ranges, bezier curves, missclick rate
  - **RL**: epsilon_start, epsilon_end, gamma, learning_rate
  - **Anti-Ban**: win_rate_target, min_matches_per_hour, max_matches_per_hour
- Botao "Restaurar Defaults"
- Botao "Exportar Config"
- Botao "Aplicar (Restart Necessario)" ou "Aplicar em Runtime" quando possivel

---

## Phase 5: Analytics e Relatorios Avancados
**Objetivo:** Entender o desempenho do bot ao longo do tempo.

### 5.1 Novas Tabs

**Tab "Mapas"**
- Win rate por mapa (tabela + grafico de barras)
- Win rate por modo de jogo
- Tempo medio de partida por mapa
- Melhor/worst brawler por mapa

**Tab "Performance"**
- Grafico de FPS ao longo do tempo
- Latencia YOLO (ms por frame)
- Tempo medio por estado (lobby, in_game, loading)
- Uso de CPU/Memoria ao longo do tempo

**Tab "RL Insights"**
- Grafico de epsilon decay
- Estados mais visitados (heatmap)
- Evolucao de Q-values por acao
- Reward acumulado por episodio

**Tab "Sessoes"**
- Historico de sessoes (data, duracao, partidas, wins, brawlers usados)
- Comparativo sessao vs sessao

**Tab "Heatmap" (futuro - requer pathfinding implementado)**
- Heatmap de mortes por posicao
- Heatmap de kills por posicao

### 5.2 Backend
- Agregar dados do ObservabilityCollector em intervalos
- Novo endpoint: `GET /api/analytics/{category}`
- Persistir agregacoes em `data/analytics/`

---

## Phase 6: Anti-Ban Dashboard
**Objetivo:** Visibilidade total do sistema anti-ban.

### 6.1 UI - Nova Tab "Anti-Ban"
- Status do AntiBanSystem (ativo/inativo)
- Win rate atual vs target
- Schedule do proximo jogo (horario randomizado)
- Padrões detetados (lista)
- Throttle status (esta a limitar?)
- Acoes de obfuscation (missclicks, delays) — contadores
- Fingerprint da sessao atual
- Configuracao de limites (editavel)

### 6.2 Endpoints
- `GET /api/antiban/status`
- `POST /api/antiban/config`

---

## Phase 7: UX Polish e Mobile
**Objetivo:** Dashboard profissional e usavel em qualquer dispositivo.

### 7.1 Melhorias de UI/UX
- **Auto-reconnect**: se servidor cair, retry exponencial
- **Toast notifications**: eventos importantes (win, loss, crash, recovery)
- **Last update timestamp**: mostra quando foi a ultima atualizacao
- **Keyboard shortcuts**: P=pause, S=start, R=restart, L=logs
- **Mobile layout**: tabs em accordion, cards empilhados
- **Dark/Light mode toggle**
- **Fullscreen screenshot**: click para expandir
- **Replay viewer basico**: sequencia de screenshots do replay

### 7.2 Performance
- **Throttling de polls**: reduzir frequencia quando tab nao esta ativa
- **Virtual scrolling**: para listas grandes (eventos, logs)
- **Compressao**: gzip no servidor HTTP

---

## Files to Modify

| File | Changes |
|------|---------|
| `pylaai_real/dashboard_server.py` | +~2000 linhas: novos endpoints, novas tabs HTML/JS, novas classes (LogBuffer integration, NotificationManager UI) |
| `wrapper.py` | Novos metodos: pause(), resume(), update_config(), set_brawler(), update_queue(), toggle_system(), execute_action() |
| `pylaai_real/state_manager.py` | Adicionar suporte a pause/resume sem matar thread |
| `core/notifications.py` | Novo ficheiro: NotificationManager |
| `core/log_buffer.py` | Novo ficheiro: LogBuffer thread-safe |

---

## Verification
- [ ] Dashboard inicia sem erros
- [ ] Todos os endpoints novos respondem corretamente
- [ ] Start/Stop/Pause/Resume funcionam pelo browser
- [ ] Logs aparecem em tempo real na tab Logs
- [ ] Notificacoes de teste funcionam
- [ ] Configuracoes sao guardadas e aplicadas
- [ ] Mobile: layout nao quebra em <500px
- [ ] Auto-reconnect funciona apos restart do servidor
- [ ] Nenhum endpoint expoe dados sensiveis (secrets, credenciais)

---

## Risks
- **Scope creep**: Este plano e grande. Recomenda-se implementar por phases.
- **Wrapper stability**: Adicionar metodos ao wrapper pode introduzir bugs no ciclo principal. Testar exaustivamente.
- **Performance**: SSE para logs pode aumentar CPU. Usar throttling.
- **Backwards compatibility**: Novos endpoints nao devem quebrar o funcionamento existente.
