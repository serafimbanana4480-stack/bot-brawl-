"""
Dashboard HTML templates (extracted from dashboard_server.py).
"""

_DASHBOARD_HTML = '''<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Soberana Omega — Dashboard</title>
<style>
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif;background:#0f172a;color:#e2e8f0;line-height:1.5}
  header{background:#1e293b;padding:1rem 1.5rem;border-bottom:1px solid #334155;display:flex;align-items:center;justify-content:space-between}
  header h1{font-size:1.25rem;color:#38bdf8}
  .status-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}
  .status-online{background:#22c55e}
  .status-offline{background:#ef4444}
  .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem;padding:1rem}
  .card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem}
  .card h2{font-size:.9rem;color:#94a3b8;text-transform:uppercase;letter-spacing:.05em;margin-bottom:.75rem}
  .metric{font-size:2rem;font-weight:700;color:#38bdf8}
  .metric.small{font-size:1.2rem}
  .label{font-size:.75rem;color:#64748b}
  .row{display:flex;justify-content:space-between;align-items:center;padding:.35rem 0;border-bottom:1px solid #334155}
  .row:last-child{border:none}
  .progress{height:6px;background:#334155;border-radius:3px;overflow:hidden;margin-top:4px}
  .progress-bar{height:100%;background:#38bdf8;border-radius:3px;transition:width .5s}
  .progress-bar.win{background:#22c55e}
  .progress-bar.loss{background:#ef4444}
  .event-log{max-height:220px;overflow-y:auto;font-family:monospace;font-size:.78rem}
  .event-log .event{padding:.2rem 0;border-bottom:1px dashed #334155}
  .event-log .time{color:#64748b}
  .btn{background:#2563eb;color:#fff;border:none;border-radius:4px;padding:.4rem .8rem;font-size:.8rem;cursor:pointer;margin-right:.4rem}
  .btn:hover{background:#1d4ed8}
  .btn.warn{background:#d97706}
  .btn.danger{background:#dc2626}
  table{width:100%;font-size:.8rem;border-collapse:collapse}
  th,td{text-align:left;padding:.4rem;border-bottom:1px solid #334155}
  th{color:#94a3b8}
  .screenshot{max-width:100%;border-radius:4px;border:1px solid #334155;margin-top:.5rem}
  .tabs{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}
  .tab{cursor:pointer;padding:.4rem .8rem;border-radius:4px;font-size:.8rem;background:#334155}
  .tab.active{background:#2563eb}
  .tab.premium{background:#7c3aed}
  .tab.premium.active{background:#9333ea}
  .tab-content{display:none}
  .tab-content.active{display:block}
  canvas{max-width:100%}
  .badge{display:inline-block;padding:.1rem .4rem;border-radius:3px;font-size:.65rem;font-weight:700;margin-left:.3rem}
  .badge.gold{background:#f59e0b;color:#000}
  /* Phase 1: Toggle Switch */
  .toggle{display:inline-block;position:relative;width:40px;height:20px}
  .toggle input{opacity:0;width:0;height:0}
  .toggle .slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;background:#334155;border-radius:20px;transition:.3s}
  .toggle .slider:before{position:absolute;content:'';height:14px;width:14px;left:3px;bottom:3px;background:#fff;border-radius:50%;transition:.3s}
  .toggle input:checked+.slider{background:#22c55e}
  .toggle input:checked+.slider:before{transform:translateX(20px)}
  /* Phase 1: Range Slider */
  input[type=range]{width:100%;margin:.5rem 0}
  /* Phase 1: Select */
  select{width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem}
  /* Phase 1: Control Panel */
  .control-row{display:flex;align-items:center;justify-content:space-between;padding:.4rem 0;border-bottom:1px solid #334155}
  .control-row:last-child{border:none}
  .control-row .label{flex:1}
  .control-row .control{flex-shrink:0;margin-left:.5rem}
  .queue-item{display:flex;align-items:center;gap:.5rem;padding:.3rem;background:#0f172a;border-radius:4px;margin-bottom:.3rem;font-size:.8rem}
  .queue-item .name{flex:1;font-weight:600;color:#38bdf8}
  .queue-item .btn-sm{padding:.2rem .4rem;font-size:.7rem;margin:0}
  .btn-sm{background:#2563eb;color:#fff;border:none;border-radius:4px;padding:.25rem .5rem;font-size:.75rem;cursor:pointer}
  .btn-sm.danger{background:#dc2626}
  .badge.silver{background:#94a3b8;color:#000}
  .badge.green{background:#22c55e;color:#000}
  .badge.red{background:#ef4444;color:#fff}
  .badge.purple{background:#7c3aed;color:#fff}
  .brawler-card{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:.75rem;margin-bottom:.5rem}
  .brawler-card .name{font-size:1rem;font-weight:700;color:#38bdf8}
  .brawler-card .stats{display:flex;gap:1rem;margin-top:.3rem;font-size:.75rem;color:#94a3b8}
  .brawler-card .stats span{display:flex;flex-direction:column;align-items:center}
  .brawler-card .stats .val{font-size:1rem;font-weight:700;color:#e2e8f0}
  .esports-bar{background:linear-gradient(90deg,#7c3aed,#2563eb);padding:.5rem 1rem;border-radius:4px;display:flex;justify-content:space-between;align-items:center;margin-bottom:.5rem}
  .esports-bar .label{font-size:.7rem;color:#e2e8f0;text-transform:uppercase;letter-spacing:.1em}
  .esports-bar .value{font-size:1.1rem;font-weight:700;color:#fff}
  .coach-tip{background:#1e293b;border-left:3px solid #7c3aed;padding:.5rem .75rem;margin-bottom:.4rem;font-size:.8rem;border-radius:0 4px 4px 0}
  .analysis-score{font-size:2.5rem;font-weight:800;text-align:center}
  .analysis-score.high{color:#22c55e}
  .analysis-score.mid{color:#f59e0b}
  .analysis-score.low{color:#ef4444}
  .win-prediction-bar{height:24px;background:#334155;border-radius:12px;overflow:hidden;position:relative}
  .win-prediction-bar .fill{height:100%;border-radius:12px;transition:width .5s}
  .win-prediction-bar .text{position:absolute;top:0;left:0;right:0;bottom:0;display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:700;color:#fff}
  .trophy-chart{background:#1e293b;border:1px solid #334155;border-radius:8px;padding:1rem}
  .weekly-card{display:flex;gap:1rem;flex-wrap:wrap}
  .weekly-card .stat{text-align:center;padding:.5rem 1rem;background:#1e293b;border:1px solid #334155;border-radius:8px;flex:1;min-width:120px}
  .weekly-card .stat .num{font-size:1.5rem;font-weight:700}
  .weekly-card .stat .lbl{font-size:.7rem;color:#94a3b8;text-transform:uppercase}
  /* Health Monitor */
  .health-dot{width:10px;height:10px;border-radius:50%;display:inline-block;margin-right:6px}
  .health-online{background:#22c55e}
  .health-warn{background:#f59e0b}
  .health-offline{background:#ef4444}
  .health-item{display:flex;align-items:center;padding:.25rem 0;font-size:.8rem}
  /* Mobile */
  @media(max-width:600px){
    .tabs{flex-wrap:wrap;gap:.3rem}
    .tab{padding:.3rem .5rem;font-size:.7rem}
    .grid{padding:.5rem;gap:.5rem}
    .card{padding:.75rem}
    .metric{font-size:1.5rem}
  }
  /* Toast */
  @keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
  /* Start button pulse animation */
  @keyframes pulse{0%{box-shadow:0 0 0 0 rgba(34,197,94,.6)}70%{box-shadow:0 0 0 10px rgba(34,197,94,0)}100%{box-shadow:0 0 0 0 rgba(34,197,94,0)}}
  .btn-start{animation:pulse 2s infinite}
  .btn-start.disabled{opacity:.5;cursor:not-allowed;animation:none}
  .btn-start.running{background:#64748b;animation:none}
  .btn.disabled{opacity:.5;cursor:not-allowed;pointer-events:none}
</style>
</head>
<body>
<header>
  <h1>Soberana Omega <span style="font-size:.75rem;color:#64748b">Dashboard</span></h1>
  <div style="display:flex;align-items:center;gap:1rem">
    <span class="btn btn-sm" onclick="exportStats()" title="Exportar estatisticas">Exportar</span>
    <span class="btn btn-sm" onclick="toggleDarkMode()" id="darkModeBtn" title="Alternar tema">Tema</span>
    <div id="connStatus"><span class="status-dot status-offline"></span>Offline</div>
  </div>
</header>

<div class="tabs" style="padding:1rem 1rem 0">
  <div class="tab active" onclick="showTab('live')">Tempo Real</div>
  <div class="tab" onclick="showTab('history')">Historico</div>
  <div class="tab" onclick="showTab('replays')">Replays</div>
  <div class="tab" onclick="showTab('abtest')">A/B Test</div>
  <div class="tab" onclick="showTab('recovery')">Recovery</div>
  <div class="tab premium" onclick="showTab('brawlers')">Brawlers</div>
  <div class="tab premium" onclick="showTab('analysis')">Match Analyzer</div>
  <div class="tab premium" onclick="showTab('aicoach')">AI Coach</div>
  <div class="tab premium" onclick="showTab('trophies')">Trophies</div>
  <div class="tab premium" onclick="showTab('esports')">Esports</div>
  <div class="tab" onclick="showTab('logs')">Logs <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('notifications')">Alertas <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('config')">Config <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('antiban')">Anti-Ban <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('analytics')">Analytics <span class="badge green">NEW</span></div>
  <div class="tab" onclick="showTab('learning')">Modo Teste <span class="badge green">LIVE</span></div>
  <div class="tab" onclick="showTab('farm')">Executar Bot <span class="badge green">LIVE</span></div>
  <div class="tab" onclick="showTab('learn')">Modo Aprender <span class="badge purple">AI</span></div>
  <div class="tab" onclick="showTab('detections')">Visao <span class="badge green">ESP</span></div>
  <div class="tab premium" onclick="showTab('training')">Training <span class="badge gold">PRO</span></div>
</div>

<div id="tab-live" class="tab-content active">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo do Bot <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="stateVal">—</div>
        <div class="label" id="brawlerVal">—</div>
        <div style="margin-top:.5rem">
          <span class="btn btn-start" id="startBtn" style="background:#22c55e" onclick="botStart()">Iniciar</span>
          <span class="btn danger disabled" id="stopBtn" onclick="botStop()">Parar</span>
          <span class="btn warn disabled" id="restartBtn" onclick="botRestart()">Reiniciar</span>
          <span class="btn disabled" id="pauseBtn" style="background:#f59e0b" onclick="botPauseToggle()">Pausar</span>
        </div>
        <div style="margin-top:.3rem">
          <span class="btn" onclick="fetch('/api/replay/start',{method:'POST',body:'{}'})">Gravar Replay</span>
          <span class="btn warn" onclick="fetch('/api/replay/stop',{method:'POST',body:'{}'})">Parar Replay</span>
          <span class="btn" style="background:#7c3aed" onclick="botAction('screenshot')">Screenshot</span>
        </div>
      </div>
      <div style="flex:1;min-width:200px">
        <div class="label" style="margin-bottom:.3rem">Selecionar Brawler</div>
        <select id="brawlerSelect" onchange="setBrawler(this.value)">
          <option value="">— Escolher —</option>
        </select>
        <div style="margin-top:.5rem">
          <span class="btn btn-sm" onclick="botAction('force_click_play')">Play</span>
          <span class="btn btn-sm" onclick="botAction('force_attack')">Atacar</span>
          <span class="btn btn-sm" onclick="botAction('force_super')">Super</span>
          <span class="btn btn-sm" onclick="botAction('force_goto_lobby')">Lobby</span>
          <span class="btn btn-sm danger" onclick="botAction('back_press')">Back</span>
        </div>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Sistemas <span class="badge green">NEW</span></h2>
    <div class="control-row">
      <span class="label">RL Engine</span>
      <label class="toggle control"><input type="checkbox" id="sysRL" onchange="toggleSystem('rl_engine',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Humanizacao</span>
      <label class="toggle control"><input type="checkbox" id="sysHuman" onchange="toggleSystem('humanization',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Anti-Ban</span>
      <label class="toggle control"><input type="checkbox" id="sysAntiBan" onchange="toggleSystem('anti_ban',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Error Recovery</span>
      <label class="toggle control"><input type="checkbox" id="sysErrRec" onchange="toggleSystem('error_recovery',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Recording</span>
      <label class="toggle control"><input type="checkbox" id="sysRec" onchange="toggleSystem('recording',this.checked)"><span class="slider"></span></label>
    </div>
    <div class="control-row">
      <span class="label">Auto-Tuner</span>
      <label class="toggle control"><input type="checkbox" id="sysTuner" onchange="toggleSystem('auto_tuner',this.checked)"><span class="slider"></span></label>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Fila de Brawlers <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="flex:2;min-width:280px">
        <div id="brawlerQueueList"><div class="label">A carregar fila...</div></div>
      </div>
      <div style="flex:1;min-width:200px">
        <h3 style="font-size:.85rem;color:#94a3b8;margin-bottom:.5rem">Adicionar Brawler</h3>
        <input id="newBrawlerName" type="text" placeholder="Nome do brawler" style="width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;margin-bottom:.3rem">
        <input id="newBrawlerTarget" type="number" placeholder="Trofeus alvo" value="350" style="width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;margin-bottom:.3rem">
        <input id="newBrawlerPriority" type="number" placeholder="Prioridade (1-5)" value="1" min="1" max="5" style="width:100%;padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;margin-bottom:.3rem">
        <span class="btn btn-sm" onclick="addBrawlerToQueue()">Adicionar</span>
        <span class="btn btn-sm warn" onclick="clearQueue()">Limpar Fila</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Health Monitor <span class="badge green">NEW</span></h2>
    <div id="healthMonitor">
      <div class="health-item"><span class="health-dot health-online"></span>YOLO Modelo: <span id="healthYOLO">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>ADB Conexao: <span id="healthADB">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>OCR: <span id="healthOCR">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>State Manager: <span id="healthState">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>RL Engine: <span id="healthRL">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>Anti-Ban: <span id="healthAntiBan">Carregando</span></div>
      <div class="health-item"><span class="health-dot health-online"></span>Emulator: <span id="healthEmulator">Carregando</span></div>
    </div>
    <div class="label" style="margin-top:.5rem">Ultima acao: <span id="healthLastAction">—</span></div>
  </div>
  <div class="card">
    <h2>Parametros de Combate <span class="badge green">NEW</span></h2>
    <div class="control-row">
      <span class="label">Aggressiveness</span>
      <span id="aggVal" class="metric small" style="width:50px;text-align:right">50%</span>
    </div>
    <input type="range" id="aggSlider" min="0" max="100" value="50" onchange="updateCombatParam('aggressiveness',this.value/100)">
    <div class="control-row">
      <span class="label">Shot Cooldown (ms)</span>
      <span id="cdVal" class="metric small" style="width:50px;text-align:right">350</span>
    </div>
    <input type="range" id="cdSlider" min="200" max="600" value="350" onchange="updateCombatParam('shot_cooldown',this.value)">
    <div class="control-row">
      <span class="label">Attack Distance</span>
      <span id="distVal" class="metric small" style="width:50px;text-align:right">200</span>
    </div>
    <input type="range" id="distSlider" min="100" max="400" value="200" onchange="updateCombatParam('attack_distance',this.value)">
  </div>
  <div class="card">
    <h2>Partidas</h2>
    <div class="metric" id="matchesVal">0</div>
    <div class="label">Win Rate: <span id="wrVal">0%</span></div>
    <div class="progress"><div class="progress-bar win" id="wrBar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <h2>Trofeus <span class="badge gold">PRO</span></h2>
    <div class="metric" id="totalTrophiesVal">0</div>
    <div class="label">Brawlers: <span id="unlockedVal">0</span>/<span id="totalBrawlersVal">80</span></div>
    <div class="progress"><div class="progress-bar" id="unlockedBar" style="width:0%;background:#7c3aed"></div></div>
  </div>
  <div class="card">
    <h2>FPS / Ciclo</h2>
    <div class="metric" id="fpsVal">0</div>
    <div class="label">Cycle: <span id="cycleVal">0</span> ms</div>
  </div>
  <div class="card">
    <h2>Combate <span class="badge purple">LIVE</span></h2>
    <div class="metric small" id="combatModeVal">neutral</div>
    <div class="label">Inimigos: <span id="enemiesVal">0</span> | HP: <span id="hpVal">100%</span></div>
    <div class="progress" style="margin-top:.3rem"><div class="progress-bar" id="hpBar" style="width:100%;background:#22c55e"></div></div>
  </div>
  <div class="card">
    <h2>Sessao</h2>
    <div class="metric small" id="uptimeVal">0:00</div>
    <div class="label">Partidas: <span id="sessionMatchesVal">0</span></div>
  </div>
  <div class="card">
    <h2>RL Q-Learning</h2>
    <div class="metric small" id="qStatesVal">0</div>
    <div class="label">Epsilon: <span id="epsVal">0.000</span></div>
    <div class="progress"><div class="progress-bar" id="epsBar" style="width:0%"></div></div>
  </div>
  <div class="card">
    <h2>ELO Combinacoes</h2>
    <div class="metric small" id="eloCountVal">0</div>
    <div class="label">Top brawlers (melhores mapas)</div>
    <div id="topElo" style="font-size:.75rem;margin-top:.3rem"></div>
  </div>
  <div class="card">
    <h2>AI Pick <span class="badge purple">PRO</span></h2>
    <div class="metric small" id="aiPickBrawler">—</div>
    <div class="label">Confianca: <span id="aiPickConf">0%</span></div>
    <div class="label" id="aiPickReason"></div>
    <div style="margin-top:.4rem">
      <div class="label">Previsao Vitoria</div>
      <div class="win-prediction-bar"><div class="fill" id="winPredBar" style="width:50%;background:#38bdf8"></div><div class="text" id="winPredText">50%</div></div>
    </div>
  </div>
  <div class="card">
    <h2>Ultimo Screenshot</h2>
    <img id="lastScreenshot" class="screenshot" src="" alt="screenshot" style="display:none">
    <div class="label" id="ssLabel">Sem screenshot</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Eventos em Tempo Real</h2>
    <div class="event-log" id="eventLog"><div class="event">A aguardar dados...</div></div>
  </div>
</div>
</div>

<div id="tab-history" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Rewards ao Longo do Tempo</h2>
    <canvas id="rewardChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Historico Completo</h2>
    <div id="historyTable"></div>
  </div>
</div>
</div>

<div id="tab-replays" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Replays Gravados</h2>
    <table><thead><tr><th>Nome</th><th>Frames</th><th>Duracao</th><th>Caminho</th></tr></thead>
    <tbody id="replayTable"><tr><td colspan="4">Carregando...</td></tr></tbody></table>
  </div>
</div>
</div>

<div id="tab-abtest" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>A/B Test Status</h2>
    <div class="metric small" id="abStatus">Inativo</div>
    <div style="margin-top:.5rem">
      <span class="btn" onclick="startAB()">Iniciar</span>
      <span class="btn danger" onclick="stopAB()">Parar</span>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Resultados</h2>
    <table><thead><tr><th>Variante</th><th>Partidas</th><th>Wins</th><th>Losses</th><th>Win Rate</th><th>Avg Reward</th></tr></thead>
    <tbody id="abTable"><tr><td colspan="6">Sem dados</td></tr></tbody></table>
  </div>
</div>
</div>

<div id="tab-recovery" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>Error Recovery</h2>
    <div class="row"><span class="label">Ativado</span><span id="erEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Erros Total</span><span id="erTotal" class="metric small">0</span></div>
    <div class="row"><span class="label">Recuperados</span><span id="erRecovered" class="metric small">0</span></div>
    <div class="row"><span class="label">Circuit Breaker</span><span id="erCircuit" class="metric small">CLOSED</span></div>
  </div>
  <div class="card">
    <h2>State Recovery</h2>
    <div class="row"><span class="label">Recovery Ativo</span><span id="srActive" class="metric small">Nao</span></div>
    <div class="row"><span class="label">Tentativas</span><span id="srAttempts" class="metric small">0</span></div>
    <div class="row"><span class="label">Estado Atual</span><span id="srState" class="metric small">—</span></div>
  </div>
  <div class="card">
    <h2>AutoCalibrator</h2>
    <div class="row"><span class="label">Ativado</span><span id="acEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Cache Size</span><span id="acCache" class="metric small">0</span></div>
  </div>
  <div class="card">
    <h2>OCR Detector</h2>
    <div class="row"><span class="label">Ativado</span><span id="ocrEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Reader Disponivel</span><span id="ocrReader" class="metric small">Nao</span></div>
  </div>
  <div class="card">
    <h2>Debug Visualizer</h2>
    <div class="row"><span class="label">Ativado</span><span id="dvEnabled" class="metric small">—</span></div>
    <div class="row"><span class="label">Em execucao</span><span id="dvRunning" class="metric small">Nao</span></div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Detalhes Recovery</h2>
    <div id="recoveryDetail" style="font-family:monospace;font-size:.78rem;max-height:200px;overflow-y:auto">
      A aguardar dados...
    </div>
  </div>
</div>
</div>

<div id="tab-brawlers" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Stats por Brawler <span class="badge gold">PRO</span></h2>
    <div id="brawlerStatsList"><div class="label">A carregar dados...</div></div>
  </div>
</div>
</div>

<div id="tab-analysis" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>Ultima Analise <span class="badge purple">PRO</span></h2>
    <div class="analysis-score mid" id="analysisScore">—</div>
    <div class="label" style="text-align:center" id="analysisResult">Sem dados</div>
  </div>
  <div class="card">
    <h2>Erros & Sugestoes</h2>
    <div id="analysisErrors" style="font-size:.8rem"><div class="label">Sem analise disponivel</div></div>
  </div>
  <div class="card">
    <h2>Pontos Fortes</h2>
    <div id="analysisStrengths" style="font-size:.8rem"><div class="label">Sem analise disponivel</div></div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Matchup & Build</h2>
    <div class="grid" style="padding:0">
      <div class="card" style="border:none">
        <h2>Matchup</h2>
        <div id="matchupAnalysis" class="label">—</div>
      </div>
      <div class="card" style="border:none">
        <h2>Build Sugerida</h2>
        <div id="buildSuggestion" class="label">—</div>
      </div>
      <div class="card" style="border:none">
        <h2>Posicionamento</h2>
        <div id="positioningTip" class="label">—</div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-aicoach" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>AI Pick Suggester <span class="badge purple">PRO</span></h2>
    <div class="metric small" id="coachPickBrawler">—</div>
    <div class="label">Mapa: <span id="coachPickMap">—</span></div>
    <div class="label">Confianca: <span id="coachPickConf">0%</span></div>
    <div class="label" id="coachPickReason"></div>
    <div class="label" style="margin-top:.3rem">Alternativas: <span id="coachPickAlts">—</span></div>
  </div>
  <div class="card">
    <h2>Previsao de Vitoria</h2>
    <div class="metric" id="coachWinPred">50%</div>
    <div class="win-prediction-bar" style="margin-top:.5rem"><div class="fill" id="coachWinBar" style="width:50%;background:#38bdf8"></div><div class="text" id="coachWinText">50%</div></div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Dicas do Coach <span class="badge purple">PRO</span></h2>
    <div id="coachTipsList"><div class="coach-tip">Joga mais partidas para receber dicas personalizadas</div></div>
  </div>
</div>
</div>

<div id="tab-trophies" class="tab-content">
<div class="grid">
  <div class="card">
    <h2>Trofeus Totais <span class="badge gold">PRO</span></h2>
    <div class="metric" id="trophyTotalVal">0</div>
    <div class="label">Brawlers desbloqueados: <span id="trophyUnlockedVal">0</span>/80</div>
  </div>
  <div class="card">
    <h2>Progresso Semanal</h2>
    <div class="weekly-card" id="weeklyProgress">
      <div class="stat"><div class="num" id="weeklyTrophies">0</div><div class="lbl">Trofeus +/-</div></div>
      <div class="stat"><div class="num" id="weeklyMatches">0</div><div class="lbl">Partidas</div></div>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Grafico de Trofeus <span class="badge gold">PRO</span></h2>
    <canvas id="trophyChart" height="250"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Evolucao Diaria</h2>
    <div id="dailyEvolutionTable"><div class="label">A carregar...</div></div>
  </div>
</div>
</div>

<div id="tab-esports" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Esports Overlay <span class="badge purple">PRO</span></h2>
    <div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);border-radius:8px;padding:1rem;border:1px solid #7c3aed">
      <!-- Top bar -->
      <div class="esports-bar">
        <div><div class="label">Brawler</div><div class="value" id="esBrawler">—</div></div>
        <div><div class="label">Estado</div><div class="value" id="esState">—</div></div>
        <div><div class="label">Mapa</div><div class="value" id="esMap">—</div></div>
        <div><div class="label">Win Rate</div><div class="value" id="esWR">0%</div></div>
      </div>
      <!-- Mid stats -->
      <div style="display:flex;gap:.5rem;margin-bottom:.5rem;flex-wrap:wrap">
        <div class="esports-bar" style="flex:1"><div><div class="label">Partidas</div><div class="value" id="esMatches">0</div></div></div>
        <div class="esports-bar" style="flex:1"><div><div class="label">Trofeus</div><div class="value" id="esTrophies">0</div></div></div>
        <div class="esports-bar" style="flex:1"><div class="label">FPS</div><div class="value" id="esFPS">0</div></div>
        <div class="esports-bar" style="flex:1"><div><div class="label">Previsao</div><div class="value" id="esPrediction">50%</div></div></div>
      </div>
      <!-- AI Coach bar -->
      <div class="esports-bar" style="background:linear-gradient(90deg,#7c3aed,#ec4899)">
        <div><div class="label">AI Pick</div><div class="value" id="esAIPick">—</div></div>
        <div><div class="label">Confianca</div><div class="value" id="esAIConf">0%</div></div>
        <div><div class="label">Razao</div><div class="value" id="esAIReason">—</div></div>
      </div>
      <!-- Coach tips scroll -->
      <div id="esCoachTips" style="font-size:.75rem;color:#c4b5fd;max-height:80px;overflow-y:auto;margin-top:.3rem">
        A aguardar dicas do coach...
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-logs" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Logs em Tempo Real <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:.5rem;flex-wrap:wrap;margin-bottom:.5rem">
      <select id="logLevel" onchange="refreshLogs()" style="width:auto">
        <option value="ALL">Todos</option>
        <option value="DEBUG">DEBUG</option>
        <option value="INFO" selected>INFO</option>
        <option value="WARNING">WARNING</option>
        <option value="ERROR">ERROR</option>
        <option value="CRITICAL">CRITICAL</option>
      </select>
      <select id="logComponent" onchange="refreshLogs()" style="width:auto">
        <option value="ALL">Todos componentes</option>
        <option value="wrapper">wrapper</option>
        <option value="state">state_manager</option>
        <option value="play">play</option>
        <option value="lobby">lobby</option>
        <option value="detect">detect</option>
        <option value="dashboard">dashboard</option>
      </select>
      <input id="logSearch" type="text" placeholder="Procurar..." oninput="refreshLogs()" style="padding:.4rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem;flex:1;min-width:150px">
      <span class="btn" onclick="refreshLogs()">Atualizar</span>
      <span class="btn" onclick="toggleLogStream()" id="logStreamBtn">Stream: OFF</span>
      <span class="btn warn" onclick="clearLogs()">Limpar</span>
    </div>
    <div id="logContainer" style="background:#0a0f1a;border:1px solid #334155;border-radius:4px;padding:.5rem;max-height:500px;overflow-y:auto;font-family:monospace;font-size:.75rem;line-height:1.6">
      <div class="label">A carregar logs...</div>
    </div>
    <div style="margin-top:.3rem;display:flex;justify-content:space-between">
      <span class="label" id="logStats">0 linhas</span>
      <label style="font-size:.75rem;color:#64748b"><input type="checkbox" id="logAutoScroll" checked> Auto-scroll</label>
    </div>
  </div>
</div>
</div>

<div id="tab-notifications" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Notificacoes e Alertas <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap">
      <div style="flex:1;min-width:280px">
        <h3 style="font-size:.85rem;color:#94a3b8;margin-bottom:.5rem">Configuracao</h3>
        <div class="control-row">
          <span class="label">Webhook URL</span>
          <input id="notifWebhook" type="text" placeholder="https://discord.com/api/webhooks/..." style="flex:1;margin-left:.5rem;padding:.3rem .5rem;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;font-size:.8rem">
        </div>
        <div class="control-row">
          <span class="label">Notificar Browser</span>
          <label class="toggle control"><input type="checkbox" id="notifBrowser" checked><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">Notificar Desktop</span>
          <label class="toggle control"><input type="checkbox" id="notifDesktop"><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">On Crash</span>
          <label class="toggle control"><input type="checkbox" id="notifCrash" checked><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">On Loss Streak (>=3)</span>
          <label class="toggle control"><input type="checkbox" id="notifLosses" checked><span class="slider"></span></label>
        </div>
        <div class="control-row">
          <span class="label">On Trophy Limit</span>
          <label class="toggle control"><input type="checkbox" id="notifTrophy" checked><span class="slider"></span></label>
        </div>
        <div style="margin-top:.5rem">
          <span class="btn" onclick="saveNotifConfig()">Guardar</span>
          <span class="btn" onclick="testNotification()">Testar</span>
        </div>
      </div>
      <div style="flex:1;min-width:280px">
        <h3 style="font-size:.85rem;color:#94a3b8;margin-bottom:.5rem">Historico</h3>
        <div id="notifHistory" style="max-height:300px;overflow-y:auto;font-size:.8rem">
          <div class="label">A carregar...</div>
        </div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-config" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Editor de Configuracao <span class="badge green">NEW</span></h2>
    <div style="display:flex;gap:.5rem;margin-bottom:.5rem;flex-wrap:wrap">
      <span class="btn" onclick="loadConfig()">Carregar</span>
      <span class="btn" onclick="saveConfig()">Guardar</span>
      <span class="btn warn" onclick="resetConfig()">Restaurar Defaults</span>
    </div>
    <textarea id="configEditor" style="width:100%;min-height:400px;background:#0f172a;color:#e2e8f0;border:1px solid #334155;border-radius:4px;padding:.5rem;font-family:monospace;font-size:.8rem" placeholder="Carregue a configuracao...">{}</textarea>
    <div id="configStatus" class="label" style="margin-top:.3rem"></div>
  </div>
</div>
</div>

<div id="tab-antiban" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Anti-Ban Dashboard <span class="badge green">NEW</span></h2>
    <div class="grid" style="padding:0">
      <div class="card" style="border:none">
        <h2>Status</h2>
        <div class="metric small" id="abStatusVal">—</div>
        <div class="label">Win Rate Target: <span id="abWinTarget">—</span></div>
        <div class="label">Win Rate Atual: <span id="abWinCurrent">—</span></div>
        <div class="label">Throttle: <span id="abThrottle">—</span></div>
      </div>
      <div class="card" style="border:none">
        <h2>Schedule</h2>
        <div class="label">Proximo jogo: <span id="abNextGame">—</span></div>
        <div class="label">Horario randomizado: <span id="abRandom">—</span></div>
      </div>
      <div class="card" style="border:none">
        <h2>Padroes Detetados</h2>
        <div id="abPatterns"><div class="label">Sem padroes</div></div>
      </div>
      <div class="card" style="border:none">
        <h2>Obfuscation</h2>
        <div class="label">Missclicks: <span id="abMissclicks">0</span></div>
        <div class="label">Delay noise: <span id="abDelayNoise">0</span></div>
        <div class="label">Fingerprint: <span id="abFingerprint">—</span></div>
      </div>
    </div>
  </div>
</div>
</div>

<div id="tab-learning" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo Modo Teste <span class="badge green">LIVE</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="lmStatusVal">Inativo</div>
        <div class="label">Partida <span id="lmMatchVal">0</span> / <span id="lmMaxVal">0</span></div>
      </div>
      <div>
        <span class="btn" style="background:#22c55e" onclick="startLearningMode()">Iniciar Modo Teste</span>
        <span class="btn danger" onclick="stopLearningMode()">Parar Modo Teste</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Kills</h2>
    <div class="metric" id="lmKillsVal">0</div>
    <div class="label">Mortes: <span id="lmDeathsVal">0</span></div>
  </div>
  <div class="card">
    <h2>Detecoes</h2>
    <div class="metric" id="lmDetectVal">0</div>
    <div class="label">Player: <span id="lmPlayerVal">0</span></div>
  </div>
  <div class="card">
    <h2>Precisao</h2>
    <div class="metric" id="lmAccuracyVal">0%</div>
    <div class="label">Kills / Ataques</div>
  </div>
  <div class="card">
    <h2>Dano</h2>
    <div class="metric small" id="lmDamageVal">0</div>
    <div class="label">Infligido</div>
  </div>
  <div class="card">
    <h2>Sobrevivencia</h2>
    <div class="metric small" id="lmSurvivalVal">0s</div>
    <div class="label">Duracao atual</div>
  </div>
  <div class="card">
    <h2>Brawler</h2>
    <div class="metric small" id="lmBrawlerVal">—</div>
    <div class="label">Em teste</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Grafico Deteccoes (ultimos 60s)</h2>
    <canvas id="lmDetectChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Kills por Partida</h2>
    <canvas id="lmKillsChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Historico de Treino</h2>
    <table><thead><tr><th>Brawler</th><th>Resultado</th><th>Kills</th><th>Mortes</th><th>Duracao</th></tr></thead>
    <tbody id="lmHistoryTable"><tr><td colspan="5">Sem dados</td></tr></tbody></table>
  </div>
</div>
</div>

<div id="tab-farm" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo Modo Executar <span class="badge green">LIVE</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="farmStatusVal">Inativo</div>
        <div class="label">Partidas: <span id="farmMatchVal">0</span> / <span id="farmTargetVal">0</span></div>
      </div>
      <div>
        <span class="btn" style="background:#22c55e" onclick="startFarmMode()">Iniciar Farm</span>
        <span class="btn danger" onclick="stopFarmMode()">Parar Farm</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Trofeus Sessao</h2>
    <div class="metric" id="farmTrophiesVal">0</div>
    <div class="label">Ganhos/perdidos</div>
  </div>
  <div class="card">
    <h2>Win Rate</h2>
    <div class="metric" id="farmWinRateVal">0%</div>
    <div class="label">W / L / D</div>
  </div>
  <div class="card">
    <h2>Tempo Medio</h2>
    <div class="metric small" id="farmAvgTimeVal">0s</div>
    <div class="label">Por partida</div>
  </div>
  <div class="card">
    <h2>APM</h2>
    <div class="metric small" id="farmApmVal">0</div>
    <div class="label">Acoes/min</div>
  </div>
  <div class="card">
    <h2>Estado</h2>
    <div class="metric small" id="farmStateVal">—</div>
    <div class="label">Atual</div>
  </div>
  <div class="card">
    <h2>Brawler</h2>
    <div class="metric small" id="farmBrawlerVal">—</div>
    <div class="label">Em uso</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Grafico de Trofeus</h2>
    <canvas id="farmTrophyChart" height="200"></canvas>
  </div>
</div>
</div>

<div id="tab-learn" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Controlo Modo Aprender <span class="badge purple">AI</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="learnStatusVal">Inativo</div>
        <div class="label">Motor: <span id="learnEngineVal">—</span></div>
      </div>
      <div>
        <span class="btn" style="background:#7c3aed" onclick="startLearnMode()">Iniciar Aprender</span>
        <span class="btn danger" onclick="stopLearnMode()">Parar Aprender</span>
      </div>
    </div>
  </div>
  <div class="card">
    <h2>Q-Table</h2>
    <div class="metric" id="learnQTableVal">0</div>
    <div class="label">Estados</div>
  </div>
  <div class="card">
    <h2>Epsilon</h2>
    <div class="metric" id="learnEpsilonVal">0.0</div>
    <div class="label">Exploracao</div>
  </div>
  <div class="card">
    <h2>Reward</h2>
    <div class="metric small" id="learnRewardVal">0</div>
    <div class="label">Ultimo / Episodio</div>
  </div>
  <div class="card">
    <h2>PPO Loss</h2>
    <div class="metric small" id="learnPpoLossVal">0</div>
    <div class="label">Policy / Value</div>
  </div>
  <div class="card">
    <h2>Buffer</h2>
    <div class="metric small" id="learnBufferVal">0</div>
    <div class="label">Experiencias</div>
  </div>
  <div class="card">
    <h2>Acao</h2>
    <div class="metric small" id="learnActionVal">—</div>
    <div class="label">Ultima escolhida</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Rewards por Episodio</h2>
    <canvas id="learnRewardChart" height="200"></canvas>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Contagem de Acoes</h2>
    <canvas id="learnActionChart" height="200"></canvas>
  </div>
</div>
</div>

<div id="tab-detections" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>ESP / Visao em Tempo Real <span class="badge green">ESP</span></h2>
    <div style="display:flex;gap:1rem;flex-wrap:wrap;align-items:center">
      <div style="flex:1;min-width:200px">
        <div class="metric small" id="espStatusVal">OFF</div>
        <div class="label">FPS: <span id="espFpsVal">0</span> | Objetos: <span id="espObjectsVal">0</span></div>
      </div>
      <div>
        <span class="btn" style="background:#22c55e" onclick="toggleESP()">Ligar ESP</span>
        <span class="btn danger" onclick="toggleESP(false)">Desligar ESP</span>
      </div>
    </div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Lista de Deteccoes</h2>
    <table><thead><tr><th>Classe</th><th>Conf</th><th>X</th><th>Y</th><th>W</th><th>H</th></tr></thead>
    <tbody id="detectionsTable"><tr><td colspan="6">Sem dados</td></tr></tbody></table>
  </div>
  <div class="card">
    <h2>Inimigos</h2>
    <div class="metric" id="detEnemyVal">0</div>
    <div class="label">Detetados</div>
  </div>
  <div class="card">
    <h2>Aliados</h2>
    <div class="metric" id="detTeamVal">0</div>
    <div class="label">Detetados</div>
  </div>
  <div class="card">
    <h2>Paredes</h2>
    <div class="metric" id="detWallVal">0</div>
    <div class="label">Obstaculos</div>
  </div>
  <div class="card">
    <h2>Arbustos</h2>
    <div class="metric" id="detBushVal">0</div>
    <div class="label">Esconderijos</div>
  </div>
  <div class="card">
    <h2>Powerups</h2>
    <div class="metric" id="detPowerVal">0</div>
    <div class="label">Itens</div>
  </div>
  <div class="card">
    <h2>Modelo</h2>
    <div class="metric small" id="detModelVal">—</div>
    <div class="label">Ativo</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Estatisticas de Visao</h2>
    <div id="visionStats">A carregar...</div>
  </div>
</div>
</div>

<div id="tab-analytics" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Analytics Avancadas <span class="badge green">NEW</span></h2>
    <div class="tabs" style="padding:0">
      <div class="tab active" onclick="showAnalyticsTab('maps')">Mapas</div>
      <div class="tab" onclick="showAnalyticsTab('perf')">Performance</div>
      <div class="tab" onclick="showAnalyticsTab('rl')">RL Insights</div>
      <div class="tab" onclick="showAnalyticsTab('sessions')">Sessoes</div>
    </div>
    <div id="analytics-maps" style="padding:1rem 0">
      <div class="label">Win Rate por Mapa (dados do Match Analyzer)</div>
      <div id="analyticsMapsContent"><div class="label">A carregar...</div></div>
    </div>
    <div id="analytics-perf" style="padding:1rem 0;display:none">
      <div class="label">FPS ao longo do tempo</div>
      <canvas id="perfFPSChart" height="200"></canvas>
      <div class="label" style="margin-top:1rem">Latencia YOLO (ms)</div>
      <div id="perfYOLO"><div class="label">N/A</div></div>
    </div>
    <div id="analytics-rl" style="padding:1rem 0;display:none">
      <div class="label">Epsilon Decay</div>
      <div id="rlEpsilonVal">Epsilon: <span class="metric small">0.000</span></div>
      <div class="label" style="margin-top:1rem">Estados Visitados</div>
      <div id="rlStatesVal">Q-States: <span class="metric small">0</span></div>
    </div>
    <div id="analytics-sessions" style="padding:1rem 0;display:none">
      <div class="label">Historico de Sessoes</div>
      <table><thead><tr><th>Inicio</th><th>Duracao</th><th>Partidas</th><th>Wins</th><th>Brawlers</th></tr></thead>
      <tbody id="sessionsTable"><tr><td colspan="5">A carregar...</td></tr></tbody></table>
    </div>
  </div>
</div>
</div>


<div id="tab-training" class="tab-content">
<div class="grid">
  <div class="card" style="grid-column:1 / -1">
    <h2>Model Registry <span class="badge gold">PRO</span></h2>
    <div style="display:flex;gap:.5rem;margin-bottom:.5rem">
      <span class="btn btn-sm" onclick="refreshTrainingModels()">Atualizar</span>
      <span class="btn btn-sm" style="background:#7c3aed" onclick="rescanModels()">Scan Models/</span>
    </div>
    <table><thead><tr><th>Modelo</th><th>Versao</th><th>Schema</th><th>mAP50</th><th>Status</th></tr></thead>
    <tbody id="modelsTable"><tr><td colspan="5">A carregar...</td></tr></tbody></table>
  </div>
  <div class="card">
    <h2>Dataset Stats</h2>
    <div class="row"><span class="label">Total Imagens</span><span id="dsImages" class="metric small">—</span></div>
    <div class="row"><span class="label">Total Boxes</span><span id="dsBoxes" class="metric small">—</span></div>
    <div class="row"><span class="label">Classes</span><span id="dsClasses" class="metric small">—</span></div>
  </div>
  <div class="card">
    <h2>Class Distribution</h2>
    <div id="dsClassDist" style="font-size:.75rem;max-height:200px;overflow-y:auto">—</div>
  </div>
  <div class="card" style="grid-column:1 / -1">
    <h2>Ultimo Treino</h2>
    <div id="lastTraining" style="font-family:monospace;font-size:.78rem;max-height:200px;overflow-y:auto">
      <div class="label">Nenhum treino registado. Execute <code>python train.py</code></div>
    </div>
  </div>
</div>
</div>

<script>
const API = '';
let lastEvents = [];

function showTab(id) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(t=>t.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('tab-'+id).classList.add('active');
}

async function poll() {
  try {
    const health = await fetch(API + '/api/health');
    if (!health.ok) throw new Error('health check failed');

    const res = await fetch(API + '/api/live');
    if (!res.ok) throw new Error('live endpoint failed');
    const d = await res.json();
    const botConnected = !!d.running || !!d.current_state || !!d.brawler;
    updateBotButtons(!!d.running);
    document.getElementById('connStatus').innerHTML = botConnected
      ? '<span class="status-dot status-online"></span>Online'
      : '<span class="status-dot status-warning"></span>Dashboard OK • Bot Offline';
    document.getElementById('stateVal').textContent = d.current_state || '—';
    document.getElementById('brawlerVal').textContent = (d.brawler || '—') + ' @ ' + (d.map_name || '—');
    document.getElementById('matchesVal').textContent = d.matches_total || 0;
    document.getElementById('wrVal').textContent = ((d.win_rate||0)*100).toFixed(1) + '%';
    document.getElementById('wrBar').style.width = ((d.win_rate||0)*100) + '%';
    document.getElementById('fpsVal').textContent = (d.fps || 0).toFixed(1);
    document.getElementById('cycleVal').textContent = (d.cycle_time_ms || 0).toFixed(1);
    document.getElementById('qStatesVal').textContent = d.q_states || 0;
    document.getElementById('epsVal').textContent = (d.epsilon || 0).toFixed(3);
    document.getElementById('epsBar').style.width = ((d.epsilon||0)/0.5*100) + '%';
    document.getElementById('eloCountVal').textContent = d.elo_combinations || 0;

    // Top ELO
    const top = d.top_elo || [];
    document.getElementById('topElo').innerHTML = top.slice(0,5).map(
      e => `<div class="row"><span>${e.brawler}@${e.map}</span><span>${e.score.toFixed(0)}</span></div>`
    ).join('') || '<div class="label">Sem dados</div>';

    // Screenshot
    if (d.screenshot_b64) {
      document.getElementById('lastScreenshot').src = 'data:image/jpeg;base64,' + d.screenshot_b64;
      document.getElementById('lastScreenshot').style.display = 'block';
      document.getElementById('ssLabel').textContent = new Date(d.timestamp*1000).toLocaleTimeString();
    }

    // Eventos
    const ev = d.recent_events || [];
    if (JSON.stringify(ev) !== JSON.stringify(lastEvents)) {
      lastEvents = ev;
      const log = document.getElementById('eventLog');
      log.innerHTML = ev.slice().reverse().map(e =>
        `<div class="event"><span class="time">${new Date(e.timestamp*1000).toLocaleTimeString()}</span> ` +
        `<strong>${e.event_type}</strong> ${JSON.stringify(e.details).slice(0,80)}</div>`
      ).join('');
    }

    // Phase 9: Recovery stats from live data
    document.getElementById('erEnabled').textContent = d.error_recovery_enabled ? 'Sim' : 'Nao';
    document.getElementById('erTotal').textContent = d.error_total || 0;
    document.getElementById('erRecovered').textContent = d.error_recovered || 0;
    document.getElementById('erCircuit').textContent = d.error_circuit_state || 'CLOSED';
    document.getElementById('srActive').textContent = d.state_recovery_active ? 'Sim' : 'Nao';
    document.getElementById('srAttempts').textContent = d.state_recovery_attempts || 0;
    document.getElementById('srState').textContent = d.state_recovery_current || '—';
    document.getElementById('acEnabled').textContent = d.autocalibrator_enabled ? 'Sim' : 'Nao';
    document.getElementById('acCache').textContent = d.autocalibrator_cache_size || 0;
    document.getElementById('ocrEnabled').textContent = d.ocr_detector_enabled ? 'Sim' : 'Nao';
    document.getElementById('ocrReader').textContent = d.ocr_reader_available ? 'Sim' : 'Nao';
    document.getElementById('dvEnabled').textContent = d.debug_visualizer_enabled ? 'Sim' : 'Nao';
    document.getElementById('dvRunning').textContent = d.debug_visualizer_running ? 'Sim' : 'Nao';

    // Premium: Trophies
    document.getElementById('totalTrophiesVal').textContent = d.total_trophies || 0;
    document.getElementById('unlockedVal').textContent = d.unlocked_brawlers || 0;
    document.getElementById('totalBrawlersVal').textContent = d.total_brawlers || 80;
    document.getElementById('unlockedBar').style.width = ((d.unlocked_brawlers||0)/(d.total_brawlers||80)*100) + '%';

    // Premium: AI Pick (live tab)
    const pick = d.ai_pick_suggestion;
    if (pick) {
      document.getElementById('aiPickBrawler').textContent = pick.brawler || '—';
      document.getElementById('aiPickConf').textContent = ((pick.confidence||0)*100).toFixed(0) + '%';
      document.getElementById('aiPickReason').textContent = pick.reason || '';
    }
    const wp = d.win_prediction || 0;
    document.getElementById('winPredBar').style.width = (wp*100) + '%';
    document.getElementById('winPredBar').style.background = wp > 0.6 ? '#22c55e' : wp > 0.4 ? '#f59e0b' : '#ef4444';
    document.getElementById('winPredText').textContent = (wp*100).toFixed(0) + '%';

    // Premium: Esports overlay
    document.getElementById('esBrawler').textContent = d.brawler || '—';
    document.getElementById('esState').textContent = d.current_state || '—';
    document.getElementById('esMap').textContent = d.map_name || '—';
    document.getElementById('esWR').textContent = ((d.win_rate||0)*100).toFixed(1) + '%';
    document.getElementById('esMatches').textContent = d.matches_total || 0;
    document.getElementById('esTrophies').textContent = d.total_trophies || 0;
    document.getElementById('esFPS').textContent = (d.fps||0).toFixed(1);
    document.getElementById('esPrediction').textContent = (wp*100).toFixed(0) + '%';
    if (pick) {
      document.getElementById('esAIPick').textContent = pick.brawler || '—';
      document.getElementById('esAIConf').textContent = ((pick.confidence||0)*100).toFixed(0) + '%';
      document.getElementById('esAIReason').textContent = pick.reason || '—';
    }
    const tips = d.coach_tips || [];
    document.getElementById('esCoachTips').innerHTML = tips.map(t => `<div>${t}</div>`).join('') || 'Sem dicas';

    // Combat & Session (real data)
    document.getElementById('combatModeVal').textContent = d.combat_mode || 'neutral';
    document.getElementById('enemiesVal').textContent = d.enemies_detected || 0;
    const hp = d.hp_estimate || 1.0;
    document.getElementById('hpVal').textContent = (hp*100).toFixed(0) + '%';
    document.getElementById('hpBar').style.width = (hp*100) + '%';
    document.getElementById('hpBar').style.background = hp > 0.6 ? '#22c55e' : hp > 0.3 ? '#f59e0b' : '#ef4444';
    // Uptime
    const uptime = d.uptime_seconds || 0;
    const hours = Math.floor(uptime/3600);
    const mins = Math.floor((uptime%3600)/60);
    const secs = Math.floor(uptime%60);
    document.getElementById('uptimeVal').textContent = hours > 0 ? `${hours}:${String(mins).padStart(2,'0')}:${String(secs).padStart(2,'0')}` : `${mins}:${String(secs).padStart(2,'0')}`;
    document.getElementById('sessionMatchesVal').textContent = d.matches_total || 0;
  } catch (err) {
    document.getElementById('connStatus').innerHTML = '<span class="status-dot status-offline"></span>Offline';
  }
}

async function pollReplays() {
  try {
    const res = await fetch(API + '/api/replays');
    const d = await res.json();
    const tbody = document.getElementById('replayTable');
    tbody.innerHTML = (d.replays || []).map(r =>
      `<tr><td>${r.name}</td><td>${r.frames}</td><td>${r.duration.toFixed(1)}s</td><td>${r.path}</td></tr>`
    ).join('') || '<tr><td colspan="4">Nenhum replay</td></tr>';
  } catch(e){}
}

async function pollAB() {
  try {
    const res = await fetch(API + '/api/abtest');
    const d = await res.json();
    document.getElementById('abStatus').textContent = d.active ? 'Ativo ('+d.current_variant+')' : 'Inativo';
    const tbody = document.getElementById('abTable');
    const vars = d.variants || {};
    tbody.innerHTML = Object.entries(vars).map(([k,v])=>
      `<tr><td>${k}</td><td>${v.matches}</td><td>${v.wins}</td><td>${v.losses}</td><td>${(v.win_rate*100).toFixed(1)}%</td><td>${v.avg_reward}</td></tr>`
    ).join('') || '<tr><td colspan="6">Sem dados</td></tr>';
  } catch(e){}
}

async function startAB() {
  await fetch(API + '/api/abtest/start', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({variants:{control:{},test_v2:{}}})
  });
  pollAB();
}
async function stopAB() {
  await fetch(API + '/api/abtest/stop', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
  pollAB();
}

// Bot control functions
async function botStart() {
  const btn = document.getElementById('startBtn');
  btn.classList.add('disabled'); btn.textContent = 'A iniciar...';
  try {
    const res = await fetch(API + '/api/bot/start', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) { toast('Bot iniciado!', 'success'); updateBotButtons(true); }
    else { toast('Erro: ' + (d.error || 'falha'), 'error'); btn.classList.remove('disabled'); btn.textContent = 'Iniciar'; }
  } catch(e) { toast('Erro ao iniciar: ' + e, 'error'); btn.classList.remove('disabled'); btn.textContent = 'Iniciar'; }
}
async function botStop() {
  const btn = document.getElementById('stopBtn');
  btn.classList.add('disabled'); btn.textContent = 'A parar...';
  try {
    const res = await fetch(API + '/api/bot/stop', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) { toast('Bot parado!', 'info'); updateBotButtons(false); }
    else { toast('Erro: ' + (d.error || 'falha'), 'error'); btn.classList.remove('disabled'); btn.textContent = 'Parar'; }
  } catch(e) { toast('Erro ao parar: ' + e, 'error'); btn.classList.remove('disabled'); btn.textContent = 'Parar'; }
}
async function botRestart() {
  if (!confirm('Reiniciar o bot?')) return;
  toast('A reiniciar bot...', 'warning', 2000);
  try {
    const res = await fetch(API + '/api/bot/restart', {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) { toast('Bot reiniciado!', 'success'); updateBotButtons(true); }
    else { toast('Erro: ' + (d.error || 'falha'), 'error'); }
  } catch(e) { toast('Erro ao reiniciar: ' + e, 'error'); }
}
function updateBotButtons(running) {
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const restartBtn = document.getElementById('restartBtn');
  const pauseBtn = document.getElementById('pauseBtn');
  if (startBtn) { startBtn.classList.toggle('disabled', running); startBtn.classList.toggle('running', running); startBtn.textContent = 'Iniciar'; }
  if (stopBtn) { stopBtn.classList.toggle('disabled', !running); stopBtn.textContent = 'Parar'; }
  if (restartBtn) { restartBtn.classList.toggle('disabled', !running); }
  if (pauseBtn) { pauseBtn.classList.toggle('disabled', !running); }
}

// Phase 1: Pause/Resume toggle
async function botPauseToggle() {
  const btn = document.getElementById('pauseBtn');
  const isPaused = btn.textContent === 'Retomar';
  try {
    const endpoint = isPaused ? '/api/bot/resume' : '/api/bot/pause';
    const res = await fetch(API + endpoint, {method:'POST',headers:{'Content-Type':'application/json'},body:'{}'});
    const d = await res.json();
    if (d.ok) {
      btn.textContent = isPaused ? 'Pausar' : 'Retomar';
      btn.style.background = isPaused ? '#f59e0b' : '#22c55e';
    } else {
      alert('Erro: ' + (d.error || 'falha'));
    }
  } catch(e) { alert('Erro: ' + e); }
}

// Phase 1: Manual actions
async function botAction(action) {
  try {
    const res = await fetch(API + '/api/bot/action', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action: action})
    });
    const d = await res.json();
    if (!d.ok) console.warn('Action failed:', d.error);
  } catch(e) { console.warn('Action error:', e); }
}

// Phase 1: Set brawler
async function setBrawler(name) {
  if (!name) return;
  try {
    const res = await fetch(API + '/api/bot/queue/set-brawler', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name: name})
    });
    const d = await res.json();
    if (d.ok) {
      document.getElementById('brawlerVal').textContent = name;
    } else {
      alert('Erro: ' + (d.error || 'falha'));
    }
  } catch(e) { alert('Erro: ' + e); }
}

// Phase 1: Toggle system
async function toggleSystem(system, enabled) {
  try {
    const res = await fetch(API + '/api/system/toggle', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({system: system, enabled: enabled})
    });
    const d = await res.json();
    if (!d.ok) console.warn('Toggle failed:', d.error);
  } catch(e) { console.warn('Toggle error:', e); }
}

async function pollSystemStatus() {
  try {
    const res = await fetch(API + '/api/system/status');
    const d = await res.json();
    if (d.systems) {
      const s = d.systems;
      const setToggle = (id, val) => { const el = document.getElementById(id); if (el) el.checked = !!val; };
      setToggle('sysRL', s.rl_engine?.enabled);
      setToggle('sysHuman', s.humanization?.enabled);
      setToggle('sysAntiBan', s.anti_ban?.enabled);
      setToggle('sysErrRec', s.error_recovery?.enabled);
      setToggle('sysRec', s.recording?.enabled);
      setToggle('sysTuner', s.auto_tuner?.enabled);
    }
    // Update pause button from bot status
    const botRes = await fetch(API + '/api/bot/status');
    const botD = await botRes.json();
    const btn = document.getElementById('pauseBtn');
    if (btn && botD.paused !== undefined) {
      btn.textContent = botD.paused ? 'Retomar' : 'Pausar';
      btn.style.background = botD.paused ? '#22c55e' : '#f59e0b';
    }
    // Update brawler dropdown
    if (botD.brawler_queue && botD.brawler_queue.length > 0) {
      const sel = document.getElementById('brawlerSelect');
      const currentVal = sel.value;
      const options = botD.brawler_queue
        .filter(b => b && (b.enabled === undefined || b.enabled))
        .map(b => `<option value="${b.name || b}">${b.name || b}</option>`).join('');
      if (sel.innerHTML.indexOf(options) === -1) {
        sel.innerHTML = '<option value="">— Escolher —</option>' + options;
        sel.value = currentVal;
      }
    }
  } catch(e) {}
}

// Phase 2: Log viewer functions
let _logStreamActive = false;
let _logStreamAbort = null;
let _logLines = [];

function _logColor(level) {
  const colors = { DEBUG: '#64748b', INFO: '#38bdf8', WARNING: '#f59e0b', ERROR: '#ef4444', CRITICAL: '#dc2626' };
  return colors[level] || '#94a3b8';
}

function _renderLogs(lines) {
  const container = document.getElementById('logContainer');
  if (!container) return;
  const autoScroll = document.getElementById('logAutoScroll')?.checked ?? true;
  container.innerHTML = lines.map(l => {
    const ts = new Date(l.timestamp * 1000).toLocaleTimeString();
    const color = _logColor(l.level);
    return `<div style="color:${color};border-bottom:1px solid #1e293b;padding:2px 0">[${ts}] <strong>${l.level}</strong> [${l.logger}] ${l.message}</div>`;
  }).join('');
  if (autoScroll) container.scrollTop = container.scrollHeight;
  document.getElementById('logStats').textContent = `${lines.length} linhas`;
}

async function refreshLogs() {
  try {
    const level = document.getElementById('logLevel')?.value || 'ALL';
    const component = document.getElementById('logComponent')?.value || 'ALL';
    const search = document.getElementById('logSearch')?.value || '';
    const params = new URLSearchParams({ limit: '200', level, component, search });
    const res = await fetch(API + '/api/logs?' + params.toString());
    const d = await res.json();
    _logLines = d.lines || [];
    _renderLogs(_logLines);
  } catch(e) {
    document.getElementById('logContainer').innerHTML = '<div class="label">Erro ao carregar logs</div>';
  }
}

function clearLogs() {
  _logLines = [];
  document.getElementById('logContainer').innerHTML = '<div class="label">Logs limpos</div>';
  document.getElementById('logStats').textContent = '0 linhas';
}

async function toggleLogStream() {
  const btn = document.getElementById('logStreamBtn');
  if (_logStreamActive) {
    _logStreamActive = false;
    if (_logStreamAbort) _logStreamAbort.abort();
    btn.textContent = 'Stream: OFF';
    btn.style.background = '#2563eb';
  } else {
    _logStreamActive = true;
    btn.textContent = 'Stream: ON';
    btn.style.background = '#22c55e';
    _startLogStream();
  }
}

async function _startLogStream() {
  try {
    const controller = new AbortController();
    _logStreamAbort = controller;
    const res = await fetch(API + '/api/logs/stream', { signal: controller.signal });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    while (_logStreamActive) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\\n\\n');
      buffer = lines.pop() || '';
      for (const chunk of lines) {
        const match = chunk.match(/^data: (.+)$/m);
        if (match) {
          try {
            const data = JSON.parse(match[1]);
            if (data.lines && data.lines.length) {
              _logLines = _logLines.concat(data.lines);
              if (_logLines.length > 500) _logLines = _logLines.slice(-500);
              _renderLogs(_logLines);
            }
          } catch(e) {}
        }
      }
    }
  } catch(e) {
    if (e.name !== 'AbortError') console.warn('Log stream error:', e);
  } finally {
    _logStreamActive = false;
    const btn = document.getElementById('logStreamBtn');
    if (btn) { btn.textContent = 'Stream: OFF'; btn.style.background = '#2563eb'; }
  }
}

async function drawRewardChart() {
  try {
    const res = await fetch(API + '/api/rewards');
    const d = await res.json();
    const pts = (d.rewards || []).slice(-60);
    const canvas = document.getElementById('rewardChart');
    if (!canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 200;
    ctx.clearRect(0,0,w,h);
    if (pts.length < 2) return;
    const vals = pts.map(p=>p.r);
    const minV = Math.min(...vals, -1), maxV = Math.max(...vals, 1);
    const range = maxV - minV || 1;
    ctx.strokeStyle = '#38bdf8'; ctx.lineWidth = 2; ctx.beginPath();
    pts.forEach((p,i)=>{
      const x = (i/(pts.length-1))*w;
      const y = h - ((p.r-minV)/range)*h;
      if(i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('Reward/frame', 4, 12);
  }catch(e){}
}

async function pollRecovery() {
  try {
    const res = await fetch(API + '/api/recovery');
    const d = await res.json();
    const detail = document.getElementById('recoveryDetail');
    if (Object.keys(d).length === 0) {
      detail.textContent = 'Sem dados de recovery disponiveis';
      return;
    }
    detail.innerHTML = Object.entries(d).map(([k,v]) =>
      `<div style="margin-bottom:.5rem"><strong>${k}:</strong> ` +
      `<pre style="margin:0;white-space:pre-wrap">${JSON.stringify(v, null, 2)}</pre></div>`
    ).join('');
  } catch(e){}
}

// Premium: Brawler stats
async function pollBrawlers() {
  try {
    const res = await fetch(API + '/api/brawlers');
    const d = await res.json();
    const list = document.getElementById('brawlerStatsList');
    const brawlers = d.brawlers || [];
    if (brawlers.length === 0) {
      list.innerHTML = '<div class="label">Sem dados de brawlers. Joga partidas para ver stats!</div>';
      return;
    }
    list.innerHTML = brawlers.sort((a,b) => b.matches - a.matches).map(b => {
      const wrColor = b.winrate >= 60 ? '#22c55e' : b.winrate >= 45 ? '#f59e0b' : '#ef4444';
      const trophyProgress = Math.min(100, (b.trophies / Math.max(1, b.target_trophies)) * 100);
      return `<div class="brawler-card">
        <div class="name">${b.name} <span class="badge ${b.winrate>=55?'green':b.winrate>=40?'silver':'red'}">${b.winrate.toFixed(1)}% WR</span></div>
        <div class="stats">
          <span><div class="val">${b.matches}</div>Picks</span>
          <span><div class="val">${b.wins}</div>Wins</span>
          <span><div class="val">${b.losses}</div>Losses</span>
          <span><div class="val" style="color:${wrColor}">${b.winrate.toFixed(1)}%</div>WR</span>
          <span><div class="val">${b.trophies}</div>Trofeus</span>
          <span><div class="val">${b.avg_kills.toFixed(1)}</div>Kills</span>
          <span><div class="val">${b.avg_deaths.toFixed(1)}</div>Deaths</span>
        </div>
        <div class="progress" style="margin-top:.3rem"><div class="progress-bar" style="width:${trophyProgress}%;background:#7c3aed"></div></div>
        <div class="label" style="margin-top:.2rem">Trofeus: ${b.trophies}/${b.target_trophies} | Melhor mapa: ${b.best_map||'—'} | Build: ${b.best_gadget||'—'}</div>
      </div>`;
    }).join('');
  } catch(e){}
}

// Premium: Match analysis
async function pollMatchAnalysis() {
  try {
    const res = await fetch(API + '/api/match-analysis');
    const d = await res.json();
    const analysis = d.match_analysis;
    if (analysis) {
      const score = analysis.score || 0;
      const scoreClass = score >= 70 ? 'high' : score >= 40 ? 'mid' : 'low';
      document.getElementById('analysisScore').textContent = score;
      document.getElementById('analysisScore').className = 'analysis-score ' + scoreClass;
      document.getElementById('analysisResult').textContent = `${analysis.brawler} @ ${analysis.map} — ${analysis.result}`;
      document.getElementById('analysisErrors').innerHTML = (analysis.errors || []).map(e =>
        `<div style="color:#ef4444;margin-bottom:.2rem">- ${e}</div>`
      ).join('') || '<div class="label">Sem erros</div>';
      document.getElementById('analysisStrengths').innerHTML = (analysis.strengths || []).map(s =>
        `<div style="color:#22c55e;margin-bottom:.2rem">+ ${s}</div>`
      ).join('') || '<div class="label">Sem pontos fortes</div>';
      document.getElementById('matchupAnalysis').textContent = analysis.matchup_analysis || '—';
      document.getElementById('buildSuggestion').textContent = analysis.build_suggestion || '—';
      document.getElementById('positioningTip').textContent = analysis.positioning_tip || '—';
    }
    // Coach tips
    const tips = d.coach_tips || [];
    document.getElementById('coachTipsList').innerHTML = tips.map(t =>
      `<div class="coach-tip">${t}</div>`
    ).join('') || '<div class="coach-tip">Joga mais partidas para receber dicas</div>';
  } catch(e){}
}

// Premium: AI Coach
async function pollAICoach() {
  try {
    const res = await fetch(API + '/api/ai-pick');
    const d = await res.json();
    const s = d.suggestion;
    if (s) {
      document.getElementById('coachPickBrawler').textContent = s.brawler || '—';
      document.getElementById('coachPickMap').textContent = s.map || d.map || '—';
      document.getElementById('coachPickConf').textContent = ((s.confidence||0)*100).toFixed(0) + '%';
      document.getElementById('coachPickReason').textContent = s.reason || '';
      document.getElementById('coachPickAlts').textContent = (s.alternatives || []).join(', ') || '—';
    }
    const wp = d.win_prediction || 0;
    document.getElementById('coachWinPred').textContent = (wp*100).toFixed(0) + '%';
    document.getElementById('coachWinPred').style.color = wp > 0.6 ? '#22c55e' : wp > 0.4 ? '#f59e0b' : '#ef4444';
    document.getElementById('coachWinBar').style.width = (wp*100) + '%';
    document.getElementById('coachWinBar').style.background = wp > 0.6 ? '#22c55e' : wp > 0.4 ? '#f59e0b' : '#ef4444';
    document.getElementById('coachWinText').textContent = (wp*100).toFixed(0) + '%';
  } catch(e){}
}

// Premium: Trophy history chart
async function drawTrophyChart() {
  try {
    const res = await fetch(API + '/api/trophy-history');
    const d = await res.json();
    const history = d.history || [];
    const canvas = document.getElementById('trophyChart');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 250;
    ctx.clearRect(0,0,w,h);
    if (history.length < 2) {
      ctx.fillStyle = '#64748b'; ctx.font = '12px sans-serif';
      ctx.fillText('Sem dados suficientes para grafico', w/2 - 120, h/2);
      return;
    }
    const vals = history.map(p => p.total_trophies);
    const minV = Math.min(...vals) - 10;
    const maxV = Math.max(...vals) + 10;
    const range = maxV - minV || 1;
    // Grid lines
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    for (let i = 0; i < 5; i++) {
      const y = (i/4) * h;
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
      ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
      ctx.fillText(Math.round(maxV - (i/4)*range), 4, y + 12);
    }
    // Trophy line
    ctx.strokeStyle = '#7c3aed'; ctx.lineWidth = 2; ctx.beginPath();
    history.forEach((p, i) => {
      const x = (i/(history.length-1))*w;
      const y = h - ((p.total_trophies - minV)/range)*h;
      if (i===0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    });
    ctx.stroke();
    // Fill under line
    ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
    ctx.fillStyle = 'rgba(124,58,237,0.1)'; ctx.fill();
    // Daily evolution table
    const daily = d.daily_evolution || [];
    const table = document.getElementById('dailyEvolutionTable');
    if (daily.length > 0) {
      table.innerHTML = '<table><thead><tr><th>Data</th><th>Trofeus</th><th>+/-</th></tr></thead><tbody>' +
        daily.slice().reverse().map(d => {
          const changeColor = d.change > 0 ? '#22c55e' : d.change < 0 ? '#ef4444' : '#94a3b8';
          const changeStr = d.change > 0 ? '+' + d.change : d.change;
          return `<tr><td>${d.date}</td><td>${d.trophies}</td><td style="color:${changeColor}">${changeStr}</td></tr>`;
        }).join('') + '</tbody></table>';
    }
  } catch(e){}
}

// Premium: Weekly progress
async function pollWeekly() {
  try {
    const res = await fetch(API + '/api/weekly-progress');
    const d = await res.json();
    const change = d.trophies_change || 0;
    const el = document.getElementById('weeklyTrophies');
    el.textContent = (change > 0 ? '+' : '') + change;
    el.style.color = change > 0 ? '#22c55e' : change < 0 ? '#ef4444' : '#94a3b8';
    document.getElementById('weeklyMatches').textContent = d.matches || 0;
    // Also update trophy tab totals
    const live = await (await fetch(API + '/api/live')).json();
    document.getElementById('trophyTotalVal').textContent = live.total_trophies || 0;
    document.getElementById('trophyUnlockedVal').textContent = live.unlocked_brawlers || 0;
  } catch(e){}
}

// Phase 3: Notifications
async function pollNotifications() {
  try {
    const res = await fetch(API + '/api/notifications/history');
    const d = await res.json();
    const container = document.getElementById('notifHistory');
    if (container) {
      const items = d.history || [];
      container.innerHTML = items.slice().reverse().map(n => {
        const color = n.level === 'error' ? '#ef4444' : n.level === 'warning' ? '#f59e0b' : '#38bdf8';
        const ts = new Date(n.timestamp * 1000).toLocaleTimeString();
        return `<div style="border-bottom:1px solid #334155;padding:.3rem 0"><span style="color:${color}"><strong>[${n.level.toUpperCase()}]</strong></span> <span style="color:#64748b">${ts}</span> ${n.title}: ${n.message}</div>`;
      }).join('') || '<div class="label">Sem notificacoes</div>';
    }
  } catch(e){}
}

async function saveNotifConfig() {
  try {
    const config = {
      webhook_url: document.getElementById('notifWebhook').value,
      browser_enabled: document.getElementById('notifBrowser').checked,
      desktop_enabled: document.getElementById('notifDesktop').checked,
      on_crash: document.getElementById('notifCrash').checked,
      on_consecutive_losses: document.getElementById('notifLosses').checked ? 3 : 0,
      on_trophy_limit: document.getElementById('notifTrophy').checked,
    };
    const res = await fetch(API + '/api/notifications/config', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(config)
    });
    const d = await res.json();
    alert(d.ok ? 'Configuracao guardada!' : 'Erro: ' + (d.error || 'falha'));
  } catch(e) { alert('Erro: ' + e); }
}

async function testNotification() {
  try {
    const res = await fetch(API + '/api/notifications/test', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({title:'Teste', message:'Notificacao de teste'})
    });
    const d = await res.json();
    alert(d.ok ? 'Notificacao de teste enviada!' : 'Erro: ' + (d.error || 'falha'));
  } catch(e) { alert('Erro: ' + e); }
}

// Phase 4: Config editor
async function loadConfig() {
  try {
    const res = await fetch(API + '/api/config');
    const d = await res.json();
    document.getElementById('configEditor').value = JSON.stringify(d, null, 2);
    document.getElementById('configStatus').textContent = 'Configuracao carregada.';
  } catch(e) { document.getElementById('configStatus').textContent = 'Erro ao carregar: ' + e; }
}

async function saveConfig() {
  try {
    const raw = document.getElementById('configEditor').value;
    const data = JSON.parse(raw);
    const res = await fetch(API + '/api/config', {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(data)
    });
    const d = await res.json();
    document.getElementById('configStatus').textContent = d.ok ? 'Configuracao guardada!' : 'Erro: ' + (d.error || 'falha');
  } catch(e) { document.getElementById('configStatus').textContent = 'Erro: ' + e; }
}

function resetConfig() {
  document.getElementById('configEditor').value = '{}';
  document.getElementById('configStatus').textContent = 'Resetado (vazio). Carregue para restaurar.';
}

// Phase 6: Anti-Ban
async function pollAntiBan() {
  try {
    const res = await fetch(API + '/api/antiban/status');
    const d = await res.json();
    document.getElementById('abStatusVal').textContent = d.enabled ? 'Ativo' : 'Inativo';
    document.getElementById('abWinTarget').textContent = (d.win_rate_target || 0.5) * 100 + '%';
    document.getElementById('abWinCurrent').textContent = ((d.current_win_rate || 0) * 100).toFixed(1) + '%';
    document.getElementById('abThrottle').textContent = d.throttling ? 'Sim' : 'Nao';
    document.getElementById('abNextGame').textContent = d.next_game_time || '—';
    document.getElementById('abRandom').textContent = d.schedule_randomized ? 'Sim' : 'Nao';
    document.getElementById('abMissclicks').textContent = d.missclicks || 0;
    document.getElementById('abDelayNoise').textContent = d.delay_noise_applied || 0;
    document.getElementById('abFingerprint').textContent = (d.fingerprint || '').slice(0, 20) + '...';
    const patterns = d.patterns_detected || [];
    document.getElementById('abPatterns').innerHTML = patterns.length ? patterns.map(p => `<div style="color:#f59e0b">- ${p}</div>`).join('') : '<div class="label">Sem padroes detetados</div>';
  } catch(e){}
}

// Learning Mode Dashboard Functions
let _lmHistory = [];

async function pollLearningMode() {
  try {
    const res = await fetch(API + '/api/learning-mode/status');
    const d = await res.json();
    const active = d.active || false;
    document.getElementById('lmStatusVal').textContent = active ? 'ATIVO' : 'Inativo';
    document.getElementById('lmStatusVal').style.color = active ? '#22c55e' : '#94a3b8';
    document.getElementById('lmMatchVal').textContent = d.current_match || 0;
    document.getElementById('lmMaxVal').textContent = d.max_matches || 0;
    document.getElementById('lmKillsVal').textContent = d.kills || 0;
    document.getElementById('lmDeathsVal').textContent = d.deaths || 0;
    document.getElementById('lmDetectVal').textContent = d.detections_enemies || 0;
    document.getElementById('lmPlayerVal').textContent = d.detections_player || 0;
    document.getElementById('lmAccuracyVal').textContent = (d.accuracy_percent || 0).toFixed(1) + '%';
    document.getElementById('lmDamageVal').textContent = (d.damage_dealt || 0).toFixed(0);
    document.getElementById('lmSurvivalVal').textContent = (d.match_duration_seconds || 0).toFixed(0) + 's';
    document.getElementById('lmBrawlerVal').textContent = d.current_brawler || '—';

    // Atualizar gráfico de detecções
    drawLearningDetectChart(d.frames_history || []);
  } catch(e) {}
}

async function pollLearningHistory() {
  try {
    const res = await fetch(API + '/api/learning-mode/history');
    const d = await res.json();
    const matches = d.matches || [];
    _lmHistory = matches;
    // Tabela
    const tbody = document.getElementById('lmHistoryTable');
    tbody.innerHTML = matches.slice().reverse().map(m =>
      `<tr><td>${m.brawler || '—'}</td><td>${m.result || '—'}</td><td>${m.kills || 0}</td><td>${m.deaths || 0}</td><td>${(m.duration_seconds || 0).toFixed(0)}s</td></tr>`
    ).join('') || '<tr><td colspan="5">Sem dados</td></tr>';
    // Gráfico kills
    drawLearningKillsChart(matches);
  } catch(e) {}
}

function drawLearningDetectChart(frames) {
  try {
    const canvas = document.getElementById('lmDetectChart');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);
    if (frames.length < 2) {
      ctx.fillStyle = '#64748b'; ctx.font = '12px sans-serif';
      ctx.fillText('A aguardar dados...', w/2 - 60, h/2);
      return;
    }
    const vals = frames.map(f => f.enemies_detected || 0);
    const maxV = Math.max(...vals, 1);
    ctx.strokeStyle = '#22c55e'; ctx.lineWidth = 2; ctx.beginPath();
    frames.forEach((f, i) => {
      const x = (i / (frames.length - 1)) * w;
      const y = h - ((f.enemies_detected || 0) / maxV) * h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.fillStyle = '#64748b'; ctx.font = '10px sans-serif';
    ctx.fillText('Inimigos detetados / frame', 4, 12);
  } catch(e) {}
}

function drawLearningKillsChart(matches) {
  try {
    const canvas = document.getElementById('lmKillsChart');
    if (!canvas || !canvas.getContext) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width = canvas.offsetWidth;
    const h = canvas.height = 200;
    ctx.clearRect(0, 0, w, h);
    if (matches.length < 1) {
      ctx.fillStyle = '#64748b'; ctx.font = '12px sans-serif';
      ctx.fillText('Sem partidas registadas', w/2 - 70, h/2);
      return;
    }
    const barW = Math.max(20, (w / matches.length) * 0.7);
    const spacing = w / matches.length;
    const maxKills = Math.max(...matches.map(m => m.kills || 0), 1);
    matches.forEach((m, i) => {
      const kills = m.kills || 0;
      const barH = (kills / maxKills) * (h - 30);
      const x = i * spacing + (spacing - barW) / 2;
      const y = h - barH - 20;
      ctx.fillStyle = kills > 0 ? '#22c55e' : '#334155';
      ctx.fillRect(x, y, barW, barH);
      ctx.fillStyle = '#94a3b8'; ctx.font = '10px sans-serif';
      ctx.fillText(kills.toString(), x + barW/2 - 4, y - 4);
      ctx.fillStyle = '#64748b'; ctx.font = '9px sans-serif';
      ctx.fillText((m.brawler || '').slice(0,4), x + 2, h - 4);
    });
  } catch(e) {}
}

async function startLearningMode() {
  try {
    const res = await fetch(API + '/api/learning-mode/start', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({max_matches: 5})
    });
    const d = await res.json();
    if (d.ok) {
      toast('Modo Teste iniciado!', 'success');
      pollLearningMode();
    } else {
      toast('Erro: ' + (d.error || 'falha'), 'error');
    }
  } catch(e) { toast('Erro: ' + e, 'error'); }
}

async function stopLearningMode() {
  if (!confirm('Parar o Modo Teste?')) return;
  try {
    const res = await fetch(API + '/api/learning-mode/stop', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: '{}'
    });
    const d = await res.json();
    if (d.ok) {
      toast('Modo Teste parado!', 'info');
      pollLearningMode();
    } else {
      toast('Erro: ' + (d.error || 'falha'), 'error');
    }
  } catch(e) { toast('Erro: ' + e, 'error'); }
}

// Phase 5: Analytics
function showAnalyticsTab(id) {
  ['maps','perf','rl','sessions'].forEach(k => {
    document.getElementById('analytics-'+k).style.display = k === id ? 'block' : 'none';
  });
}

async function pollAnalytics() {
  try {
    const live = await (await fetch(API + '/api/live')).json();
    // Maps tab: use brawler stats
    const maps = await (await fetch(API + '/api/brawlers')).json();
    if (maps.brawlers) {
      const html = maps.brawlers.slice(0,10).map(b => {
        return `<div class="row"><span>${b.name}</span><span>${b.winrate.toFixed(1)}% WR (${b.matches} matches)</span></div>`;
      }).join('');
      document.getElementById('analyticsMapsContent').innerHTML = html || '<div class="label">Sem dados</div>';
    }
    // RL tab
    document.getElementById('rlEpsilonVal').innerHTML = 'Epsilon: <span class="metric small">' + (live.epsilon || 0).toFixed(3) + '</span>';
    document.getElementById('rlStatesVal').innerHTML = 'Q-States: <span class="metric small">' + (live.q_states || 0) + '</span>';
  } catch(e){}
}

// Phase 1: Brawler Queue UI
async function pollQueue() {
  try {
    const res = await fetch(API + '/api/bot/queue');
    const d = await res.json();
    const container = document.getElementById('brawlerQueueList');
    if (!container) return;
    const items = d.queue || [];
    if (items.length === 0) {
      container.innerHTML = '<div class="label">Fila vazia. Adicione brawlers abaixo.</div>';
      return;
    }
    container.innerHTML = items.map((b, idx) => {
      const isCurrent = b.current ? '<span class="badge green">ATIVO</span>' : '';
      const trophyProg = Math.min(100, (b.current_trophies / Math.max(1, b.target_trophies)) * 100);
      return `<div class="queue-item" style="${b.current ? 'border:1px solid #22c55e' : ''}">
        <span class="name">${b.name} ${isCurrent}</span>
        <span class="label">${b.current_trophies}/${b.target_trophies} trofeus</span>
        <span class="label">P:${b.priority}</span>
        <span class="btn-sm" onclick="moveQueueItem(${idx},-1)" ${idx===0?'style="visibility:hidden"':''}>&uarr;</span>
        <span class="btn-sm" onclick="moveQueueItem(${idx},1)" ${idx===items.length-1?'style="visibility:hidden"':''}>&darr;</span>
        <span class="btn-sm danger" onclick="removeBrawlerFromQueue(${idx})">x</span>
      </div>`;
    }).join('');
  } catch(e) {}
}

async function addBrawlerToQueue() {
  const name = document.getElementById('newBrawlerName').value.trim();
  const target = parseInt(document.getElementById('newBrawlerTarget').value) || 350;
  const priority = parseInt(document.getElementById('newBrawlerPriority').value) || 1;
  if (!name) { alert('Insira um nome de brawler'); return; }
  try {
    const res = await fetch(API + '/api/bot/queue/add', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({name, target_trophies: target, priority})
    });
    const d = await res.json();
    if (d.ok) {
      document.getElementById('newBrawlerName').value = '';
      pollQueue();
      toast(`${name} adicionado a fila!`, 'success');
    } else {
      alert('Erro: ' + (d.error || 'falha'));
    }
  } catch(e) { alert('Erro: ' + e); }
}

async function removeBrawlerFromQueue(index) {
  try {
    const res = await fetch(API + '/api/bot/queue/remove', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({index})
    });
    const d = await res.json();
    if (d.ok) { pollQueue(); toast('Brawler removido', 'info'); }
  } catch(e) {}
}

async function moveQueueItem(index, direction) {
  try {
    const res = await fetch(API + '/api/bot/queue');
    const d = await res.json();
    const queue = d.queue || [];
    const newIndex = index + direction;
    if (newIndex < 0 || newIndex >= queue.length) return;
    const temp = queue[index];
    queue[index] = queue[newIndex];
    queue[newIndex] = temp;
    // Rebuild queue data for update
    const updateRes = await fetch(API + '/api/bot/queue/update', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({queue: queue.map(b => ({
        name: b.name, current_trophies: b.current_trophies,
        target_trophies: b.target_trophies, priority: b.priority, enabled: b.enabled
      }))})
    });
    if (updateRes.ok) pollQueue();
  } catch(e) {}
}

async function clearQueue() {
  if (!confirm('Limpar toda a fila de brawlers?')) return;
  try {
    const res = await fetch(API + '/api/bot/queue/update', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({queue: []})
    });
    if (res.ok) { pollQueue(); toast('Fila limpa', 'info'); }
  } catch(e) {}
}

// Health Monitor
async function pollHealth() {
  try {
    const res = await fetch(API + '/api/bot/status');
    const d = await res.json();
    const setHealth = (id, status, text) => {
      const el = document.getElementById(id);
      if (!el) return;
      const dot = el.parentElement.querySelector('.health-dot');
      el.textContent = text;
      if (dot) {
        dot.className = 'health-dot ' + (status === 'ok' ? 'health-online' : status === 'warn' ? 'health-warn' : 'health-offline');
      }
    };
    setHealth('healthYOLO', d.models_loaded ? 'ok' : 'error', d.models_loaded ? 'Carregado' : 'Nao carregado');
    setHealth('healthADB', d.emulator_controller_active ? 'ok' : 'error', d.emulator_controller_active ? 'Conectado' : 'Desconectado');
    setHealth('healthOCR', d.ocr_detector?.reader_available ? 'ok' : 'warn', d.ocr_detector?.reader_available ? 'Disponivel' : 'Indisponivel');
    setHealth('healthState', d.running ? 'ok' : 'warn', d.current_state || 'unknown');
    setHealth('healthRL', d.systems?.rl_engine?.enabled ? 'ok' : 'warn', d.systems?.rl_engine?.enabled ? 'Ativo' : 'Inativo');
    setHealth('healthAntiBan', d.systems?.anti_ban?.enabled ? 'ok' : 'warn', d.systems?.anti_ban?.enabled ? 'Ativo' : 'Inativo');
    setHealth('healthEmulator', d.window_active ? 'ok' : 'warn', d.window_active ? 'Janela ativa' : 'Janela inativa');
    document.getElementById('healthLastAction').textContent = d.session_duration ? (d.session_duration / 60).toFixed(1) + ' min' : '—';
  } catch(e) {}
}

// Combat params
async function updateCombatParam(param, value) {
  try {
    const res = await fetch(API + '/api/bot/combat/param', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({param, value: parseFloat(value)})
    });
    const d = await res.json();
    if (d.ok) {
      if (param === 'aggressiveness') document.getElementById('aggVal').textContent = Math.round(value*100) + '%';
      if (param === 'shot_cooldown') document.getElementById('cdVal').textContent = value;
      if (param === 'attack_distance') document.getElementById('distVal').textContent = value;
      toast(param + ' atualizado: ' + value, 'success', 1500);
    }
  } catch(e) { console.warn('Combat param error:', e); }
}

// Phase 7: Export stats
async function exportStats() {
  try {
    const res = await fetch(API + '/api/export/stats');
    const d = await res.json();
    const blob = new Blob([JSON.stringify(d, null, 2)], {type: 'application/json'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `soberana_stats_${new Date().toISOString().slice(0,10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast('Estatisticas exportadas!', 'success');
  } catch(e) { toast('Erro ao exportar', 'error'); }
}

// Phase 7: Dark/Light mode
let _darkMode = true;
function toggleDarkMode() {
  _darkMode = !_darkMode;
  const btn = document.getElementById('darkModeBtn');
  const root = document.documentElement;
  if (_darkMode) {
    root.style.setProperty('--bg', '#0f172a');
    root.style.setProperty('--card-bg', '#1e293b');
    root.style.setProperty('--text', '#e2e8f0');
    root.style.setProperty('--text-muted', '#94a3b8');
    root.style.setProperty('--border', '#334155');
    root.style.setProperty('--input-bg', '#0f172a');
    document.body.style.background = '#0f172a';
    document.body.style.color = '#e2e8f0';
    if (btn) btn.textContent = 'Light';
  } else {
    root.style.setProperty('--bg', '#f8fafc');
    root.style.setProperty('--card-bg', '#ffffff');
    root.style.setProperty('--text', '#1e293b');
    root.style.setProperty('--text-muted', '#64748b');
    root.style.setProperty('--border', '#e2e8f0');
    root.style.setProperty('--input-bg', '#f1f5f9');
    document.body.style.background = '#f8fafc';
    document.body.style.color = '#1e293b';
    if (btn) btn.textContent = 'Dark';
  }
}

// Phase 7: UX Polish
// Toast notifications
function toast(message, type='info', duration=3000) {
  const container = document.getElementById('toastContainer') || (() => {
    const el = document.createElement('div');
    el.id = 'toastContainer';
    el.style.cssText = 'position:fixed;bottom:1rem;right:1rem;z-index:9999;display:flex;flex-direction:column;gap:.5rem';
    document.body.appendChild(el);
    return el;
  })();
  const toast = document.createElement('div');
  const colors = { info:'#2563eb', success:'#22c55e', warning:'#f59e0b', error:'#ef4444' };
  toast.style.cssText = `background:${colors[type]||colors.info};color:#fff;padding:.6rem 1rem;border-radius:4px;font-size:.85rem;box-shadow:0 4px 12px rgba(0,0,0,.3);animation:fadeIn .3s;max-width:300px`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity='0'; toast.style.transition='opacity .3s'; setTimeout(()=>toast.remove(),300); }, duration);
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'p' || e.key === 'P') botPauseToggle();
  if (e.key === 's' || e.key === 'S') botStart();
  if (e.key === 'r' || e.key === 'R') { e.preventDefault(); botRestart(); }
  if (e.key === 'l' || e.key === 'L') showTab('logs');
});

// Auto-reconnect indicator
let _lastPollSuccess = true;
let _reconnectAttempts = 0;
const originalPoll = poll;
poll = async function() {
  try {
    await originalPoll();
    if (!_lastPollSuccess) {
      _lastPollSuccess = true;
      _reconnectAttempts = 0;
      toast('Dashboard reconectada!', 'success', 2000);
    }
  } catch(e) {
    _lastPollSuccess = false;
    _reconnectAttempts++;
    const status = document.getElementById('connStatus');
    if (status) status.innerHTML = '<span class="status-dot status-offline"></span>Offline (' + _reconnectAttempts + ')';
  }
};

// Mode Control Center Functions
async function startFarmMode() { try { const res=await fetch(API+'/api/mode/farm/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:{}})}); const d=await res.json(); toast(d.status==='started'?'Farm iniciado':'Falha ao iniciar farm',d.ok?'success':'error'); } catch(e){ toast('Erro: '+e,'error'); } }
async function stopFarmMode() { try { const res=await fetch(API+'/api/mode/farm/stop',{method:'POST',headers:{'Content-Type':'application/json'}}); const d=await res.json(); toast(d.status==='stopped'?'Farm parado':'Falha','info'); } catch(e){} }
async function startLearnMode() { try { const res=await fetch(API+'/api/mode/learn/start',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({config:{}})}); const d=await res.json(); toast(d.status==='started'?'Aprender iniciado':'Falha',d.ok?'success':'error'); } catch(e){ toast('Erro: '+e,'error'); } }
async function stopLearnMode() { try { const res=await fetch(API+'/api/mode/learn/stop',{method:'POST',headers:{'Content-Type':'application/json'}}); const d=await res.json(); toast(d.status==='stopped'?'Aprender parado':'Falha','info'); } catch(e){} }
async function toggleESP(force) { try { const enabled = force !== undefined ? force : true; const res=await fetch(API+'/api/esp/toggle',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({enabled})}); const d=await res.json(); toast('ESP '+d.status,d.ok?'success':'error'); } catch(e){} }

async function pollModeStatus() {
  try {
    const res=await fetch(API+'/api/mode/status'); const d=await res.json();
    // Farm
    const farmActive = d.active_mode==='farm';
    document.getElementById('farmStatusVal').textContent = farmActive?'ATIVO':'Inativo';
    document.getElementById('farmStatusVal').style.color = farmActive?'#22c55e':'#94a3b8';
    document.getElementById('farmMatchVal').textContent = d.matches_completed||0;
    document.getElementById('farmTargetVal').textContent = d.matches_target||0;
    document.getElementById('farmBrawlerVal').textContent = d.current_brawler||'—';
    // Learn
    const learnActive = d.active_mode==='learn';
    document.getElementById('learnStatusVal').textContent = learnActive?'ATIVO':'Inativo';
    document.getElementById('learnStatusVal').style.color = learnActive?'#7c3aed':'#94a3b8';
  } catch(e){}
}

async function pollRLMetrics() {
  try {
    const res=await fetch(API+'/api/rl/metrics'); const d=await res.json();
    document.getElementById('learnEngineVal').textContent = d.engine_type||'—';
    document.getElementById('learnQTableVal').textContent = d.q_table_size||0;
    document.getElementById('learnEpsilonVal').textContent = (d.epsilon||0).toFixed(3);
    document.getElementById('learnRewardVal').textContent = (d.last_reward||0).toFixed(2)+' / '+(d.episode_reward||0).toFixed(2);
    document.getElementById('learnPpoLossVal').textContent = (d.policy_loss||0).toFixed(4)+' / '+(d.value_loss||0).toFixed(4);
    document.getElementById('learnBufferVal').textContent = (d.buffer_size||0)+' / '+(d.buffer_capacity||0);
    document.getElementById('learnActionVal').textContent = d.last_action||'—';
  } catch(e){}
}

async function pollDetections() {
  try {
    const res=await fetch(API+'/api/detections/live'); const d=await res.json();
    const detections = d.detections||[];
    // ESP
    document.getElementById('espFpsVal').textContent = (d.fps||0).toFixed(1);
    document.getElementById('espObjectsVal').textContent = detections.length;
    // Table
    const tbody = document.getElementById('detectionsTable');
    if (!detections.length) { tbody.innerHTML='<tr><td colspan="6">Sem dados</td></tr>'; }
    else { tbody.innerHTML=detections.slice(0,20).map(det=>`<tr><td>${det.class_name}</td><td>${(det.confidence||0).toFixed(2)}</td><td>${det.x}</td><td>${det.y}</td><td>${det.width}</td><td>${det.height}</td></tr>`).join(''); }
    // Counters
    const counts={}; detections.forEach(d=>counts[d.class_name]=(counts[d.class_name]||0)+1);
    document.getElementById('detEnemyVal').textContent = counts['enemy']||0;
    document.getElementById('detTeamVal').textContent = counts['teammate']||0;
    document.getElementById('detWallVal').textContent = counts['wall']||0;
    document.getElementById('detBushVal').textContent = counts['bush']||0;
    document.getElementById('detPowerVal').textContent = counts['powerup']||0;
  } catch(e){}
}

async function pollVisionStats() {
  try {
    const res=await fetch(API+'/api/vision/stats'); const d=await res.json();
    document.getElementById('detModelVal').textContent = (d.device||'—')+' / '+(d.models_loaded||0)+' modelos';
    const statsDiv = document.getElementById('visionStats');
    statsDiv.innerHTML = '<div class="row"><span>Initialized</span><span>'+d.initialized+'</span></div>'
      +'<div class="row"><span>Frame count</span><span>'+(d.frame_count||0)+'</span></div>'
      +'<div class="row"><span>Avg confidence</span><span>'+((d.avg_confidence||0).toFixed(3))+'</span></div>'
      +'<div class="row"><span>Loaded classes</span><span>'+(d.loaded_classes?d.loaded_classes.join(', '):'—')+'</span></div>';
  } catch(e){}
}

// CSS animation for toasts
const toastStyle = document.createElement('style');
toastStyle.textContent = '@keyframes fadeIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}';
document.head.appendChild(toastStyle);


async function refreshTrainingModels() {
  try {
    const res = await fetch(API + '/api/training/models');
    const d = await res.json();
    const tbody = document.getElementById('modelsTable');
    const models = d.models || [];
    if (!models.length) {
      tbody.innerHTML = '<tr><td colspan="5">Nenhum modelo registado. Execute <code>python train.py</code></td></tr>';
    } else {
      tbody.innerHTML = models.map(m =>
        '<tr><td>' + (m.name || '—') + '</td><td>' + (m.version || '—') + '</td><td>' + (m.schema || '—') + '</td><td>' + ((m.map50 || 0).toFixed(3)) + '</td><td>' + (m.is_active ? '<span style="color:#22c55e">Ativo</span>' : '<span style="color:#64748b">Inativo</span>') + '</td></tr>'
      ).join('');
    }
    // Dataset
    document.getElementById('dsImages').textContent = d.dataset?.total_images || '—';
    document.getElementById('dsBoxes').textContent = d.dataset?.total_boxes || '—';
    document.getElementById('dsClasses').textContent = d.dataset?.num_classes || '—';
    const dist = d.dataset?.class_distribution || {};
    document.getElementById('dsClassDist').innerHTML = Object.entries(dist).map(([k, v]) =>
      '<div class="row"><span>' + k + '</span><span>' + v + '</span></div>'
    ).join('') || '—';
    // Last training
    const lt = d.last_training;
    if (lt) {
      document.getElementById('lastTraining').innerHTML =
        '<div class="row"><span>Run ID</span><span>' + (lt.run_id || '—') + '</span></div>' +
        '<div class="row"><span>Data</span><span>' + (lt.timestamp || '—') + '</span></div>' +
        '<div class="row"><span>Schema</span><span>' + (lt.schema || '—') + '</span></div>' +
        '<div class="row"><span>mAP50</span><span>' + ((lt.map50 || 0).toFixed(4)) + '</span></div>' +
        '<div class="row"><span>mAP50-95</span><span>' + ((lt.map50_95 || 0).toFixed(4)) + '</span></div>' +
        '<div class="row"><span>Duracao</span><span>' + (lt.duration_seconds || 0).toFixed(0) + 's</span></div>';
    }
  } catch(e) {}
}
async function rescanModels() {
  try {
    const res = await fetch(API + '/api/training/models?scan=1');
    const d = await res.json();
    toast(d.scanned + ' modelos encontrados', 'info');
    refreshTrainingModels();
  } catch(e) { toast('Erro ao scan', 'error'); }
}

setInterval(poll, 2000);
setInterval(pollReplays, 5000);
setInterval(pollAB, 5000);
setInterval(pollRecovery, 5000);
setInterval(drawRewardChart, 3000);
setInterval(pollBrawlers, 5000);
setInterval(pollMatchAnalysis, 5000);
setInterval(pollAICoach, 5000);
setInterval(drawTrophyChart, 5000);
setInterval(pollWeekly, 10000);
setInterval(pollSystemStatus, 3000);
setInterval(refreshLogs, 5000);
setInterval(pollNotifications, 10000);
setInterval(pollAntiBan, 10000);
setInterval(pollAnalytics, 10000);
setInterval(pollQueue, 5000);
setInterval(pollHealth, 3000);
setInterval(pollLearningMode, 2000);
setInterval(pollLearningHistory, 10000);
setInterval(pollModeStatus, 2000);
setInterval(pollRLMetrics, 2000);
setInterval(pollDetections, 1000);
setInterval(pollVisionStats, 5000);
setInterval(refreshTrainingModels, 15000);
poll(); pollReplays(); pollAB(); pollRecovery(); drawRewardChart();
pollBrawlers(); pollMatchAnalysis(); pollAICoach(); drawTrophyChart(); pollWeekly(); pollSystemStatus(); refreshLogs(); pollNotifications(); pollAntiBan(); pollAnalytics(); pollQueue(); pollHealth(); pollLearningMode(); pollLearningHistory();
pollModeStatus(); pollRLMetrics(); pollDetections(); pollVisionStats(); refreshTrainingModels();
</script>
</body>
</html>
'''

# ---------------------------------------------------------------------------
# DASHBOARD SERVER (orquestrador)
# ---------------------------------------------------------------------------

