
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
      const lines = buffer.split('\n\n');
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
