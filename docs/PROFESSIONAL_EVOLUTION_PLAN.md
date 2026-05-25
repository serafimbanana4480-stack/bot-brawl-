# PROFESSIONAL EVOLUTION PLAN - Soberana Omega Brawl Stars Bot

**Data:** 2026-05-17  
**Objetivo:** Transformar o bot de nível "funcional" para "profissional máximo"  
**Horizonte:** 12 meses (dividido em 4 fases de 3 meses)  
**Filosofia:** Qualidade > Quantidade, Robustez > Features, Autonomia > Automação

---

## 📊 AUDITORIA COMPLETA DO ESTADO ATUAL

### ✅ O Que Já Foi Feito (Base Sólida)

| Categoria | Estado | Qualidade |
|-----------|--------|-----------|
| **Módulos Decorativos** | ✅ Removidos (7 módulos, ~3.350 linhas) | Excelente |
| **RL Engine** | ✅ Funcional com HP real | Bom |
| **adaptive_screenshot** | ✅ Integrado com TTL adaptativo | Bom |
| **Dataset YOLO** | ✅ 8 classes preparado | Bom |
| **Training Pipeline** | ✅ Pipeline híbrido Roboflow + custom | Excelente |
| **Combat Advanced v2** | ✅ Leading shot, kiting, cover, combos | Excelente |
| **Humanization Utils** | ✅ Jitter, Bezier, APM, reaction delays | Bom |
| **State Detection** | ✅ 10 estados + smoothing/voting | Bom |
| **Lobby Navigator v2** | ✅ Popups, smart play button, fast brawler select | Bom |
| **Dashboard** | ✅ Real-time web dashboard | Bom |
| **A/B Testing** | ✅ Framework implementado | Bom |
| **Replay Recorder** | ✅ 150 frames per replay | Bom |

### ⚠️ O Que Falta (Gaps Críticos)

| Área | Gap Crítico | Impacto |
|------|-------------|---------|
| **Perceção** | Sem memória espacial persistente | Muito Alto |
| **Perceção** | Calibração de confiança ausente | Alto |
| **Perceção** | Detecção multi-camada não implementada | Alto |
| **Decisão** | Sem árvore de decisão contextual | Muito Alto |
| **Decisão** | Sem planeamento multi-passo | Alto |
| **Decceção** | Sem teoria da mente (prever inimigo) | Alto |
| **Combate** | Sem gestão de recursos como economia | Alto |
| **Combate** | Sem adaptação por brawler específica | Médio |
| **Anti-Ban** | Perfil comportamental estático | Muito Alto |
| **Anti-Ban** | Sem curva de aprendizagem simulada | Alto |
| **Anti-Ban** | Sem micro-comportamentos humanos | Alto |
| **Resiliência** | Sem hierarquia de recovery | Muito Alto |
| **Resiliência** | Sem circuit breakers | Alto |
| **Resiliência** | Sem graceful degradation | Alto |
| **Observabilidade** | Sem metrics pipeline completo | Alto |
| **Observabilidade** | Sem tracing distribuído | Médio |
| **Arquitetura** | God Objects não refatorados (419KB) | Alto |
| **Arquitetura** | Sem sistema de plugins | Médio |
| **Arquitetura** | Sem feature flags | Médio |
| **Evolução** | Sem active learning loop | Muito Alto |
| **Evolução** | Sem self-improvement | Muito Alto |
| **Evolução** | Sem experimentação autónoma | Alto |

---

## 🎯 FASE 1: RESILIÊNCIA E ROBUSTEZ (Meses 1-3)

**Filosofia:** De nada serve ser inteligente se crasha a cada 30 minutos.  
**Objetivo:** O bot nunca para por causa de erros inesperados.

### 1.1 Hierarquia de Recovery

**Problema Atual:** O bot crasha em estados desconhecidos.  
**Solução:** Implementar 5 níveis de recovery automático.

```python
class RecoveryHierarchy:
    def __init__(self):
        self.level_1_attempts = 0  # Tentar resolver estado
        self.level_2_attempts = 0  # Voltar atrás
        self.level_3_attempts = 0  # Reset suave
        self.level_4_attempts = 0  # Reset duro
        self.level_5_escalated = False  # Escalar para humano
    
    def handle_error(self, error_context):
        """Tenta recovery em ordem de custo crescente"""
        if self._try_level_1(error_context):
            return
        if self._try_level_2(error_context):
            return
        if self._try_level_3(error_context):
            return
        if self._try_level_4(error_context):
            return
        self._escalate_to_human(error_context)
```

**Implementação:**
- Criar `core/recovery_hierarchy.py`
- Integrar com `state_manager.py` e `wrapper.py`
- Adicionar logging detalhado de cada tentativa
- Métricas: recovery_success_rate por nível

**Entregáveis:**
- [ ] `core/recovery_hierarchy.py` (~300 linhas)
- [ ] Integração com `state_manager.py`
- [ ] Integração com `wrapper.py`
- [ ] Testes unitários (90% cobertura)
- [ ] Documentação no AGENTS.md

### 1.2 Circuit Breakers

**Problema Atual:** O bot entra em loops infinitos quando algo falha.  
**Solução:** Implementar circuit breakers por subsistema.

```python
class CircuitBreaker:
    def __init__(self, subsystem_name, threshold=5, timeout=60):
        self.name = subsystem_name
        self.failure_count = 0
        self.threshold = threshold
        self.timeout = timeout
        self.last_failure_time = None
        self.state = "closed"  # closed, open, half_open
    
    def call(self, func, *args, **kwargs):
        """Executa função com proteção de circuit breaker"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "half_open"
            else:
                raise CircuitBreakerOpen(f"{self.name} circuit is open")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

**Circuit Breakers Necessários:**
- **Deteção:** Se >50% dos frames não têm deteções durante 30s
- **Combate:** Se não há mudança de posição há 10s
- **Lobby:** Se tenta clicar "Play" 5x e falha
- **Emulador:** Se o screenshot é sempre igual (crash do jogo)

**Implementação:**
- Criar `core/circuit_breaker.py`
- Adicionar circuit breakers a cada subsistema crítico
- Dashboard mostra estado de cada circuit breaker
- Auto-reset após timeout configurável

**Entregáveis:**
- [ ] `core/circuit_breaker.py` (~250 linhas)
- [ ] Integração com deteção, combate, lobby, emulador
- [ ] Dashboard integration (circuit breaker status)
- [ ] Testes unitários (85% cobertura)

### 1.3 Watchdogs Independentes

**Problema Atual:** Se o loop principal parar, ninguém sabe.  
**Solução:** Thread separada que monitora o loop principal.

```python
class Watchdog:
    def __init__(self, heartbeat_timeout=60):
        self.heartbeat_timeout = heartbeat_timeout
        self.last_heartbeat = time.time()
        self.alert_callbacks = []
    
    def heartbeat(self):
        """Loop principal chama isto periodicamente"""
        self.last_heartbeat = time.time()
    
    def start_monitoring(self):
        """Thread separada monitora heartbeat"""
        while True:
            time.sleep(5)
            if time.time() - self.last_heartbeat > self.heartbeat_timeout:
                self._trigger_alert("Main loop unresponsive")
                self._attempt_recovery()
```

**Watchdogs Necessários:**
- Main loop watchdog (60s timeout)
- Emulador connection watchdog (30s timeout)
- Screenshot capture watchdog (20s timeout)

**Implementação:**
- Criar `core/watchdog.py`
- Thread separada com baixo overhead
- Alertas via dashboard + logging
- Recovery automático (restart loop, reconectar ADB)

**Entregáveis:**
- [ ] `core/watchdog.py` (~200 linhas)
- [ ] Integração com wrapper.py
- [ ] Dashboard integration (watchdog status)
- [ ] Testes unitários (80% cobertura)

### 1.4 Graceful Degradation

**Problema Atual:** Se algo falha parcialmente, o bot crasha.  
**Solução:** Degradação graciosa quando subsistemas falham.

```python
class GracefulDegradation:
    def __init__(self):
        self.degradation_levels = {
            "full": {"detection": "yolo", "combat": "advanced"},
            "degraded_1": {"detection": "yolo", "combat": "basic"},
            "degraded_2": {"detection": "template", "combat": "basic"},
            "minimal": {"detection": "template", "combat": "passive"}
        }
        self.current_level = "full"
    
    def degrade(self, failed_subsystem):
        """Degrada nível se subsistema crítico falhar"""
        if failed_subsystem == "yolo_detection":
            self.current_level = "degraded_2"
        elif failed_subsystem == "combat_advanced":
            self.current_level = "degraded_1"
```

**Cenários de Degradação:**
- YOLO crashou → Usa template matching básico
- OCR falhou → Usa heurísticas de cor para estimar HP
- Emulador lento → Aumenta intervalo entre capturas
- Perda de conexão → Entra em modo offline, tenta reconectar

**Implementação:**
- Criar `core/graceful_degradation.py`
- Configurar fallbacks por subsistema
- Dashboard mostra nível de degradação
- Auto-recovery quando subsistema volta

**Entregáveis:**
- [ ] `core/graceful_degradation.py` (~250 linhas)
- [ ] Fallbacks implementados para todos subsistemas
- [ ] Dashboard integration (degradation level)
- [ ] Testes unitários (85% cobertura)

### 1.5 State Persistence

**Problema Atual:** Se o bot parar por qualquer razão, perde-se o progresso.  
**Solução:** Guardar o estado atual periodicamente.

```python
class StatePersistence:
    def __init__(self, state_file="data/bot_state.json"):
        self.state_file = state_file
        self.state = {}
    
    def save_state(self, key, value):
        """Guarda estado persistente"""
        self.state[key] = value
        self._persist_to_disk()
    
    def load_state(self, key, default=None):
        """Recupera estado persistente"""
        return self.state.get(key, default)
    
    def _persist_to_disk(self):
        """Escreve estado para disco"""
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)
```

**Estado Persistente:**
- Trophies atuais
- Brawler selecionado
- Partidas jogadas na sessão
- Win rate da sessão
- Último mapa jogado
- ELO ratings por brawler+map
- Q-table do RL engine

**Implementação:**
- Criar `core/state_persistence.py`
- Auto-save a cada 30s ou após cada partida
- Auto-load ao iniciar
- Atomic writes (evitar corrupção)

**Entregáveis:**
- [ ] `core/state_persistence.py` (~200 linhas)
- [ ] Integração com wrapper.py, rl_engine.py, elo_tracker.py
- [ ] Testes unitários (80% cobertura)

---

## 🎯 FASE 2: PERCEÇÃO MULTI-CAMADA (Meses 4-6)

**Filosofia:** Ver melhor = jogar melhor, e o fundamento de tudo.  
**Objetivo:** Sistema de visão com 3 camadas + memória espacial.

### 2.1 Deteção Multi-Camada

**Problema Atual:** Um único YOLO para tudo.  
**Solução:** 3 camadas de deteção com especialização.

```python
class MultiLayerDetection:
    def __init__(self):
        self.layer_1 = YOLO("yolov8n.pt")  # Rápido, 30fps
        self.layer_2 = YOLO("yolov8s.pt")  # Preciso, 10fps
        self.layer_3 = OCREngine()  # OCR dedicado
        self.layer_4 = TemplateMatcher()  # Elementos fixos
    
    def detect(self, frame):
        """Pipeline multi-camada"""
        # Camada 1: Detecção grossa rápida
        rough_detections = self.layer_1(frame, conf=0.3)
        
        # Camada 2: Verificação precisa em ROIs
        precise_detections = self._verify_with_layer_2(frame, rough_detections)
        
        # Camada 3: OCR para números
        numbers = self.layer_3.extract_numbers(frame)
        
        # Camada 4: Template matching para UI
        ui_elements = self.layer_4.match_templates(frame)
        
        return self._merge_all(precise_detections, numbers, ui_elements)
```

**Camadas:**
- **Camada 1:** YOLO nano para deteção grossa a 30fps
- **Camada 2:** YOLO small/medium para verificação a 10fps
- **Camada 3:** OCR dedicado para números (HP, timer, trophies)
- **Camada 4:** Template matching para elementos fixos (botões, icones)

**Implementação:**
- Criar `vision/multi_layer_detection.py`
- Integrar com `detect.py`
- Pipeline paralelo (async)
- Métricas: FPS por camada, accuracy trade-off

**Entregáveis:**
- [ ] `vision/multi_layer_detection.py` (~400 linhas)
- [ ] Integração com detect.py
- [ ] OCR engine para HP/timer
- [ ] Template matcher para UI
- [ ] Testes unitários (85% cobertura)

### 2.2 Memória Espacial Persistente

**Problema Atual:** O modelo só vê o frame atual. Um humano lembra-se.  
**Solução:** Occupancy grid com decay temporal.

```python
class SpatialMemory:
    def __init__(self, grid_size=100):
        self.grid_size = grid_size
        self.occupancy_grid = np.zeros((grid_size, grid_size))
        self.decay_rate = 0.95  # Perde 5% por segundo
        self.last_update = time.time()
    
    def update(self, detections):
        """Atualiza grid com deteções atuais"""
        # Decay temporal
        dt = time.time() - self.last_update
        self.occupancy_grid *= (self.decay_rate ** dt)
        
        # Adicionar novas deteções
        for detection in detections:
            grid_x, grid_y = self._world_to_grid(detection.position)
            self.occupancy_grid[grid_y, grid_x] = 1.0
        
        self.last_update = time.time()
    
    def query(self, position, radius=5):
        """Query: o que está nesta zona?"""
        grid_x, grid_y = self._world_to_grid(position)
        region = self.occupancy_grid[grid_y-radius:grid_y+radius, 
                                      grid_x-radius:grid_x+radius]
        return region.sum()  # Soma de probabilidade
```

**Memória Espacial:**
- "Aquele bush tinha um inimigo há 2 segundos"
- "O power cube estava ali atrás da parede"
- "O inimigo foi para a direita, provavelmente vai fazer flank"

**Implementação:**
- Criar `core/spatial_memory.py` (já existe, expandir)
- Grid 100x100 com decay temporal
- Query por posição + raio
- Integrar com combat e movement

**Entregáveis:**
- [ ] Expandir `core/spatial_memory.py` (+200 linhas)
- [ ] Integração com combat_advanced.py
- [ ] Integração com movement.py
- [ ] Visualização no dashboard (heatmap)
- [ ] Testes unitários (80% cobertura)

### 2.3 Predição de Trajetória com Física

**Problema Atual:** Navegação reativa, não preditiva.  
**Solução:** Predição de movimento baseada em física.

```python
class TrajectoryPredictor:
    def __init__(self):
        self.velocity_filter = ExponentialMovingAverage(alpha=0.3)
        self.physics_model = BrawlStarsPhysics()
    
    def predict_position(self, enemy, time_horizon=1.0):
        """Prediz posição do inimigo em time_horizon segundos"""
        # Filtrar velocidade (suavizar zig-zags)
        velocity = self.velocity_filter.update(enemy.velocity)
        
        # Se parado, não predizer movimento
        if velocity.magnitude() < 10:
            return enemy.position
        
        # Predizer posição com física
        predicted = enemy.position + velocity * time_horizon
        
        # Clamp ao mapa (não pode sair dos limites)
        predicted = self.physics_model.clamp_to_map(predicted)
        
        return predicted
```

**Física Implementada:**
- Velocidade do brawler (pixels/s)
- Limites do mapa
- Obstáculos (paredes)
- Aceleração e desaceleração

**Implementação:**
- Criar `vision/trajectory_predictor.py`
- Integrar com leading shot engine
- Integrar com kiting engine
- Calibração por brawler

**Entregáveis:**
- [ ] `vision/trajectory_predictor.py` (~300 linhas)
- [ ] Integração com combat_advanced.py
- [ ] Brawler-specific physics
- [ ] Testes unitários (85% cobertura)

### 2.4 Calibração de Confiança

**Problema Atual:** O modelo diz "87% confiança" mas não é calibrado.  
**Solução:** Calibração de confiança com temperatura scaling.

```python
class ConfidenceCalibrator:
    def __init__(self):
        self.temperature = 1.0
        self.calibration_data = []
    
    def calibrate(self, predictions, ground_truth):
        """Calibra temperatura usando validation set"""
        # Coletar (confiança, acerto) para cada predição
        for pred, gt in zip(predictions, ground_truth):
            self.calibration_data.append((pred.confidence, pred.iou > 0.5))
        
        # Otimizar temperatura
        self.temperature = self._optimize_temperature()
    
    def apply(self, confidence):
        """Aplica calibração a uma confiança"""
        calibrated = confidence ** (1.0 / self.temperature)
        return np.clip(calibrated, 0.0, 1.0)
```

**Calibração:**
- Se modelo diz 87% 100 vezes, 87 devem estar certas
- Temperature scaling para ajustar
- Re-calibrar periodicamente com novos dados

**Implementação:**
- Criar `vision/confidence_calibrator.py`
- Dataset de calibração (validation set)
- Re-calibração automática a cada X partidas
- Dashboard mostra calibration curve

**Entregáveis:**
- [ ] `vision/confidence_calibrator.py` (~250 linhas)
- [ ] Dataset de calibração
- [ ] Re-calibração automática
- [ ] Dashboard integration (calibration curve)
- [ ] Testes unitários (80% cobertura)

---

## 🎯 FASE 3: ANTI-BAN COMPORTAMENTAL (Meses 7-9)

**Filosofia:** Podes ter o melhor bot do mundo, se for banido acabou.  
**Objetivo:** Impercetível a qualquer sistema anti-cheat.

### 3.1 Perfil Comportamental Dinâmico

**Problema Atual:** Perfil estático, detetável.  
**Solução:** Cada sessão tem um perfil aleatório único.

```python
class BehavioralProfile:
    def __init__(self):
        self.age = random.randint(16, 35)  # Refletido nos reflexos
        self.experience = random.uniform(0.3, 0.9)  # Skill level
        self.aggression = random.uniform(0.2, 0.8)  # Passivo vs agressivo
        self.patience = random.uniform(0.3, 0.9)  # Tolerância a espera
        self.risk_tolerance = random.uniform(0.2, 0.8)
    
    def get_reaction_delay(self, base_delay):
        """Ajusta delay baseado no perfil"""
        age_factor = self.age / 35.0  # Mais velho = mais lento
        exp_factor = 1.0 - self.experience  # Menos exp = mais lento
        return base_delay * age_factor * exp_factor
    
    def get_aggression_modifier(self):
        """Modificador de agressividade"""
        return self.aggression
```

**Perfis Dinâmicos:**
- Sessão 1: Jovem, agressivo, impaciente
- Sessão 2: Velho, passivo, paciente
- Sessão 3: Médio, equilibrado
- Cada sessão é "um jogador diferente"

**Implementação:**
- Criar `core/behavioral_profile.py` (já existe, expandir)
- Gerar perfil ao iniciar sessão
- Persistir perfil durante sessão
- Aplicar a todas as decisões

**Entregáveis:**
- [ ] Expandir `core/behavioral_profile.py` (+150 linhas)
- [ ] Integração com humanization_utils.py
- [ ] Integração com combat_advanced.py
- [ ] Integração com decision system
- [ ] Testes unitários (80% cobertura)

### 3.2 Curva de Aprendizagem Simulada

**Problema Atual:** Performance constante, não natural.  
**Solução:** Simular aquecimento, peak, e fadiga.

```python
class LearningCurve:
    def __init__(self):
        self.session_start = time.time()
        self.peak_time = 15 * 60  # 15 minutos para peak
        self.fatigue_start = 45 * 60  # 45 minutos para começar fadiga
    
    def get_performance_modifier(self):
        """Retorna modificador de performance (0.0-1.0)"""
        elapsed = time.time() - self.session_start
        
        if elapsed < self.peak_time:
            # Aquecimento: 0.6 → 1.0
            return 0.6 + 0.4 * (elapsed / self.peak_time)
        elif elapsed < self.fatigue_start:
            # Peak: 1.0
            return 1.0
        else:
            # Fadiga: 1.0 → 0.7
            fatigue = (elapsed - self.fatigue_start) / (60 * 60)  # 1h para cair a 0.7
            return max(0.7, 1.0 - 0.3 * fatigue)
```

**Curva de Aprendizagem:**
- Início: Mais erros, reações mais lentas (a "aquecer")
- Meio: Peak performance
- Fim: Fadiga, mais erros, decisões piores

**Implementação:**
- Criar `core/learning_curve.py`
- Aplicar a: reaction delays, precision, decision quality
- Sessões longas têm degradação mais acentuada
- Dashboard mostra curva de performance

**Entregáveis:**
- [ ] `core/learning_curve.py` (~200 linhas)
- [ ] Integração com humanization_utils.py
- [ ] Integração com combat_advanced.py
- [ ] Dashboard integration (performance curve)
- [ ] Testes unitários (80% cobertura)

### 3.3 Micro-comportamentos Humanos

**Problema Atual:** Movimentos perfeitos, não naturais.  
**Solução:** Adicionar imperfeições humanas.

```python
class MicroBehaviors:
    def __init__(self):
        self.over_correction_chance = 0.15
        self.hesitation_chance = 0.10
        self.tunnel_vision_chance = 0.08
        self.panic_movement_chance = 0.05
        self.victory_spin_chance = 0.30
        self.emote_chance = 0.05
        self.accidental_touch_chance = 0.02
    
    def apply_over_correction(self, target):
        """Over-correction: mira demasiado para um lado, depois corrige"""
        if random.random() < self.over_correction_chance:
            offset = random.uniform(10, 30)  # pixels
            direction = random.choice([-1, 1])
            return target + (offset * direction)
        return target
    
    def apply_hesitation(self):
        """Hesitation: demora 100-300ms a reagir"""
        if random.random() < self.hesitation_chance:
            return random.uniform(0.1, 0.3)
        return 0.0
```

**Micro-comportamentos:**
- Over-correction: Miras demasiado para um lado
- Hesitation: Demora a reagir
- Tunnel vision: Ignora inimigos brevemente
- Panic movements: Movimentos erraticos quando toma dano
- Victory spins: Spin de celebracao
- Emote usage: Usa emotes em momentos sociais
- Accidental touches: Toca no ecra sem querer

**Implementação:**
- Criar `core/micro_behaviors.py`
- Aplicar a todos os inputs e decisões
- Probabilidades configuráveis
- Dashboard mostra estatísticas

**Entregáveis:**
- [ ] `core/micro_behaviors.py` (~300 linhas)
- [ ] Integração com emulator_controller.py
- [ ] Integração com combat_advanced.py
- [ ] Integração com humanization_utils.py
- [ ] Testes unitários (75% cobertura)

### 3.4 Padrões de Sessão Realistas

**Problema Atual:** Padrões de jogo não naturais.  
**Solução:** Simular padrões humanos de sessão.

```python
class SessionPatterns:
    def __init__(self):
        self.play_hours = self._generate_play_hours()
        self.session_duration = self._generate_duration()
        self.break_duration = self._generate_break()
        self.off_days = self._generate_off_days()
    
    def _generate_play_hours(self):
        """Horarios de jogo variaveis com padrao"""
        base_hour = random.choice([18, 19, 20, 21])  # Mais a noite
        variation = random.uniform(-1, 1)
        return [(base_hour + variation) % 24]
    
    def _generate_duration(self):
        """Duracao de sessao: 30min a 2h"""
        return random.uniform(30, 120)  # minutos
    
    def _generate_break(self):
        """Pausas naturais: 5-15 min"""
        return random.uniform(5, 15)  # minutos
```

**Padrões de Sessão:**
- Horários variáveis mas com padrão (mais à noite)
- Duração variável (30min a 2h)
- Pausas naturais (5-15min entre sessões)
- Dias sem jogar (1-2 dias por semana)
- Modos variados (não só o mesmo modo)

**Implementação:**
- Criar `core/session_patterns.py`
- Wrapper respeita padrões
- Auto-pausa após duração configurada
- Auto-resumo após break
- Dashboard mostra padrões

**Entregáveis:**
- [ ] `core/session_patterns.py` (~250 linhas)
- [ ] Integração com wrapper.py
- [ ] Auto-pause/resume
- [ ] Dashboard integration (session patterns)
- [ ] Testes unitários (80% cobertura)

---

## 🎯 FASE 4: EVOLUÇÃO AUTÓNOMA (Meses 10-12)

**Filosofia:** O objetivo final: o bot melhora sozinho, sem intervenção humana.  
**Objetivo:** Self-improvement loop completo.

### 4.1 Active Learning Loop

**Problema Atual:** Modelo estático, não aprende.  
**Solução:** Loop de active learning automático.

```python
class ActiveLearningLoop:
    def __init__(self):
        self.uncertainty_threshold = 0.5
        self.uncertain_frames = []
        self.human_review_queue = []
    
    def collect_uncertain_frames(self, frame, detections):
        """Guarda frames com deteções incertas"""
        for detection in detections:
            if detection.confidence < self.uncertainty_threshold:
                self.uncertain_frames.append((frame, detection))
        
        # Manter apenas os 1000 mais incertos
        self.uncertain_frames = self.uncertain_frames[-1000:]
    
    def request_human_review(self):
        """Solicita revisao humana dos frames mais incertos"""
        # Selecionar 100 frames mais incertos
        top_uncertain = self.uncertain_frames[:100]
        self.human_review_queue.extend(top_uncertain)
        
        # Limpar queue
        self.uncertain_frames = []
    
    def retrain_with_new_labels(self, labels):
        """Retreina modelo com novos labels"""
        # Adicionar labels ao dataset
        self._add_labels_to_dataset(labels)
        
        # Retreinar modelo
        new_model = self._train_model()
        
        # Validar novo modelo
        if self._validate_model(new_model):
            self._deploy_model(new_model)
```

**Active Learning Loop:**
1. O bot joga partidas
2. Guarda frames com baixa confiança
3. Humano (ou sistema semi-automatico) rotula esses frames
4. Modelo é re-treinado focando nesses casos difíceis
5. Ciclo repete-se

**Implementação:**
- Criar `training/active_learning_loop.py`
- Interface de labeling simples
- Retreino automático
- A/B testing de novos modelos

**Entregáveis:**
- [ ] `training/active_learning_loop.py` (~400 linhas)
- [ ] Interface de labeling (web)
- [ ] Retreino automático
- [ ] A/B testing integration
- [ ] Testes unitários (75% cobertura)

### 4.2 Self-Improvement Loop

**Problema Atual:** Não há auto-melhoria.  
**Solução:** Loop de self-improvement automático.

```python
class SelfImprovementLoop:
    def __init__(self):
        self.analysis_interval = 100  # Analisar a cada 100 partidas
        self.improvement_threshold = 0.05  # 5% de melhoria
    
    def analyze_performance(self, recent_matches):
        """Analisa as partidas mais recentes"""
        # Identificar as 20 piores partidas
        worst_matches = sorted(recent_matches, key=lambda m: m.reward)[:20]
        
        # Analisar padrões de falha
        failure_patterns = self._extract_failure_patterns(worst_matches)
        
        return failure_patterns
    
    def generate_improvement_plan(self, failure_patterns):
        """Gera plano de melhoria"""
        plan = []
        for pattern in failure_patterns:
            if pattern.type == "detection":
                plan.append(("collect_data", pattern.context))
            elif pattern.type == "decision":
                plan.append(("adjust_parameters", pattern.context))
            elif pattern.type == "combat":
                plan.append(("train_model", pattern.context))
        
        return plan
```

**Self-Improvement Loop:**
1. Joga 100 partidas
2. Analisa as 20 piores
3. Identifica padrões de falha
4. Gera ou coleta dados para esse cenário
5. Re-treina modelo com novos dados
6. Valida que novo modelo é melhor
7. Faz deploy do novo modelo
8. Volta ao passo 1

**Implementação:**
- Criar `training/self_improvement_loop.py`
- Análise automática de partidas
- Geração de planos de melhoria
- Execução automática de planos

**Entregáveis:**
- [ ] `training/self_improvement_loop.py` (~350 linhas)
- [ ] Análise de partidas
- [ ] Geração de planos
- [ ] Execução automática
- [ ] Testes unitários (70% cobertura)

### 4.3 Meta-Learning

**Problema Atual:** Não há aprendizado de aprendizado.  
**Solução:** O bot aprende a aprender.

```python
class MetaLearning:
    def __init__(self):
        self.strategies = {}  # map -> strategy -> performance
        self.matchups = {}  # brawler_pair -> win_rate
    
    def learn_strategy_performance(self, map, strategy, performance):
        """Aprende performance de estrategia por mapa"""
        if map not in self.strategies:
            self.strategies[map] = {}
        if strategy not in self.strategies[map]:
            self.strategies[map][strategy] = []
        
        self.strategies[map][strategy].append(performance)
    
    def get_best_strategy(self, map):
        """Retorna melhor estrategia para mapa"""
        if map not in self.strategies:
            return "default"
        
        strategies = self.strategies[map]
        best = max(strategies.items(), key=lambda x: np.mean(x[1]))
        return best[0]
    
    def learn_matchup(self, my_brawler, enemy_brawler, result):
        """Aprende matchups"""
        pair = f"{my_brawler}_vs_{enemy_brawler}"
        if pair not in self.matchups:
            self.matchups[pair] = []
        
        self.matchups[pair].append(result)  # 1 = win, 0 = loss
```

**Meta-Learning:**
- "Neste mapa, brawlers de longo alcance têm 20% mais win rate"
- "Contra este brawler específico, jogar agressivo resulta em morte 80% das vezes"
- "Power cubes no centro do mapa são armadilhas 70% das vezes"
- Regras descobertas automaticamente

**Implementação:**
- Criar `decision/meta_learning.py` (já existe, expandir)
- Aprendizado de estratégias por mapa
- Aprendizado de matchups
- Dashboard mostra descobertas

**Entregáveis:**
- [ ] Expandir `decision/meta_learning.py` (+200 linhas)
- [ ] Integração com decision system
- [ ] Integração com rl_engine.py
- [ ] Dashboard integration (meta-learning insights)
- [ ] Testes unitários (75% cobertura)

### 4.4 Experimentação Autónoma

**Problema Atual:** Não há experimentação.  
**Solução:** O bot testa hipóteses automaticamente.

```python
class AutonomousExperimentation:
    def __init__(self):
        self.hypotheses = []
        self.experiments = {}
    
    def generate_hypothesis(self):
        """Gera hipotese para testar"""
        hypothesis_types = [
            "wait_at_start",
            "aggressive_vs_defensive",
            "cube_collection_strategy",
            "super_timing",
            "retreat_threshold"
        ]
        
        h_type = random.choice(hypothesis_types)
        hypothesis = Hypothesis(h_type, self._generate_params(h_type))
        self.hypotheses.append(hypothesis)
        return hypothesis
    
    def run_experiment(self, hypothesis):
        """Executa experimento"""
        experiment_id = str(uuid.uuid4())
        self.experiments[experiment_id] = {
            "hypothesis": hypothesis,
            "results": [],
            "start_time": time.time()
        }
        
        # Executar N partidas com hipotese
        for _ in range(20):
            result = self._play_match_with_hypothesis(hypothesis)
            self.experiments[experiment_id]["results"].append(result)
        
        return self._analyze_results(experiment_id)
```

**Experimentação Autónoma:**
- "Sera que esperar 5s no inicio melhora a sobrevivencia?"
- "Sera que atacar o inimigo mais proximo e melhor que o com menos HP?"
- O sistema varia parametros e mede resultados
- Hipoteses confirmadas tornam-se regras permanentes

**Implementação:**
- Criar `decision/autonomous_experimentation.py`
- Geração automática de hipóteses
- Execução automática de experimentos
- Análise automática de resultados

**Entregáveis:**
- [ ] `decision/autonomous_experimentation.py` (~400 linhas)
- [ ] Geração de hipóteses
- [ ] Execução de experimentos
- [ ] Análise de resultados
- [ ] Testes unitários (70% cobertura)

---

## 📋 PRIORIZAÇÃO POR IMPACTO VS ESFORÇO

| Melhoria | Impacto | Esforço | Prioridade | Fase |
|----------|---------|--------|------------|------|
| **Hierarquia de Recovery** | Muito Alto | Médio | 🔴 P0 | Fase 1 |
| **Circuit Breakers** | Muito Alto | Médio | 🔴 P0 | Fase 1 |
| **Watchdogs** | Alto | Baixo | 🔴 P0 | Fase 1 |
| **Graceful Degradation** | Muito Alto | Médio | 🔴 P0 | Fase 1 |
| **State Persistence** | Alto | Baixo | 🔴 P0 | Fase 1 |
| **Memória Espacial** | Muito Alto | Médio | 🟡 P1 | Fase 2 |
| **Deteção Multi-Camada** | Alto | Alto | 🟡 P1 | Fase 2 |
| **Predição Trajetória** | Alto | Médio | 🟡 P1 | Fase 2 |
| **Calibração Confiança** | Médio | Médio | 🟢 P2 | Fase 2 |
| **Perfil Comportamental** | Muito Alto | Médio | 🔴 P0 | Fase 3 |
| **Curva Aprendizagem** | Alto | Baixo | 🟡 P1 | Fase 3 |
| **Micro-comportamentos** | Alto | Médio | 🟡 P1 | Fase 3 |
| **Padrões de Sessão** | Alto | Médio | 🟡 P1 | Fase 3 |
| **Active Learning Loop** | Muito Alto | Muito Alto | 🟡 P1 | Fase 4 |
| **Self-Improvement Loop** | Muito Alto | Muito Alto | 🟡 P1 | Fase 4 |
| **Meta-Learning** | Alto | Médio | 🟢 P2 | Fase 4 |
| **Experimentação Autónoma** | Alto | Alto | 🟢 P2 | Fase 4 |

**Legenda:**
- 🔴 P0: Crítico - fazer o mais cedo possível
- 🟡 P1: Importante - fazer após P0
- 🟢 P2: Desejável - fazer se houver tempo

---

## 📅 CRONOGRAMA DETALHADO

### Mês 1-3: Fase 1 - Resiliência e Robustez

**Semana 1-2:**
- [ ] Implementar `core/recovery_hierarchy.py`
- [ ] Integrar com state_manager.py
- [ ] Integrar com wrapper.py
- [ ] Testes unitários (90% cobertura)

**Semana 3-4:**
- [ ] Implementar `core/circuit_breaker.py`
- [ ] Adicionar circuit breakers a todos subsistemas
- [ ] Dashboard integration
- [ ] Testes unitários (85% cobertura)

**Semana 5-6:**
- [ ] Implementar `core/watchdog.py`
- [ ] Thread separada de monitoring
- [ ] Dashboard integration
- [ ] Testes unitários (80% cobertura)

**Semana 7-8:**
- [ ] Implementar `core/graceful_degradation.py`
- [ ] Fallbacks para todos subsistemas
- [ ] Dashboard integration
- [ ] Testes unitários (85% cobertura)

**Semana 9-10:**
- [ ] Implementar `core/state_persistence.py`
- [ ] Integração com wrapper, rl_engine, elo_tracker
- [ ] Testes unitários (80% cobertura)

**Semana 11-12:**
- [ ] Integração completa Fase 1
- [ ] Testes E2E
- [ ] Documentação
- [ ] Release Fase 1

### Mês 4-6: Fase 2 - Perceção Multi-Camada

**Semana 13-14:**
- [ ] Implementar `vision/multi_layer_detection.py`
- [ ] Integração com detect.py
- [ ] OCR engine para HP/timer
- [ ] Testes unitários (85% cobertura)

**Semana 15-16:**
- [ ] Expandir `core/spatial_memory.py`
- [ ] Integração com combat_advanced.py
- [ ] Integração com movement.py
- [ ] Visualização no dashboard

**Semana 17-18:**
- [ ] Implementar `vision/trajectory_predictor.py`
- [ ] Integração com leading shot engine
- [ ] Integração com kiting engine
- [ ] Testes unitários (85% cobertura)

**Semana 19-20:**
- [ ] Implementar `vision/confidence_calibrator.py`
- [ ] Dataset de calibração
- [ ] Re-calibração automática
- [ ] Dashboard integration

**Semana 21-22:**
- [ ] Integração completa Fase 2
- [ ] Testes E2E
- [ ] Documentação
- [ ] Release Fase 2

### Mês 7-9: Fase 3 - Anti-Ban Comportamental

**Semana 23-24:**
- [ ] Expandir `core/behavioral_profile.py`
- [ ] Integração com humanization_utils.py
- [ ] Integração com combat_advanced.py
- [ ] Testes unitários (80% cobertura)

**Semana 25-26:**
- [ ] Implementar `core/learning_curve.py`
- [ ] Integração com humanization_utils.py
- [ ] Integração com combat_advanced.py
- [ ] Dashboard integration

**Semana 27-28:**
- [ ] Implementar `core/micro_behaviors.py`
- [ ] Integração com emulator_controller.py
- [ ] Integração com combat_advanced.py
- [ ] Testes unitários (75% cobertura)

**Semana 29-30:**
- [ ] Implementar `core/session_patterns.py`
- [ ] Integração com wrapper.py
- [ ] Auto-pause/resume
- [ ] Dashboard integration

**Semana 31-32:**
- [ ] Integração completa Fase 3
- [ ] Testes E2E
- [ ] Documentação
- [ ] Release Fase 3

### Mês 10-12: Fase 4 - Evolução Autónoma

**Semana 33-36:**
- [ ] Implementar `training/active_learning_loop.py`
- [ ] Interface de labeling (web)
- [ ] Retreino automático
- [ ] A/B testing integration

**Semana 37-40:**
- [ ] Implementar `training/self_improvement_loop.py`
- [ ] Análise de partidas
- [ ] Geração de planos
- [ ] Execução automática

**Semana 41-42:**
- [ ] Expandir `decision/meta_learning.py`
- [ ] Integração com decision system
- [ ] Integração com rl_engine.py
- [ ] Dashboard integration

**Semana 43-44:**
- [ ] Implementar `decision/autonomous_experimentation.py`
- [ ] Geração de hipóteses
- [ ] Execução de experimentos
- [ ] Análise de resultados

**Semana 45-48:**
- [ ] Integração completa Fase 4
- [ ] Testes E2E
- [ ] Documentação
- [ ] Release Fase 4

---

## 📈 MÉTRICAS DE SUCESSO

### Fase 1: Resiliência
- **Uptime:** >95% (atualmente ~70%)
- **MTTR (Mean Time To Recovery):** <30s (atualmente ~5min)
- **Crashes por 100 partidas:** <1 (atualmente ~10)
- **Recovery success rate:** >90% (atualmente ~50%)

### Fase 2: Perceção
- **mAP50:** >75% (atualmente ~60%)
- **FPS médio:** >25 (atualmente ~15)
- **False positive rate:** <5% (atualmente ~15%)
- **Spatial memory hit rate:** >70% (novo)

### Fase 3: Anti-Ban
- **Detection risk score:** <10% (novo)
- **Behavioral similarity score:** >80% (novo)
- **Session pattern realism:** >85% (novo)
- **Ban rate:** 0% (manter)

### Fase 4: Evolução
- **Win rate improvement:** +15% em 6 meses (novo)
- **Self-improvement cycles:** 1 por semana (novo)
- **Meta-learning rules:** >50 descobertas (novo)
- **Experimentation success rate:** >30% (novo)

---

## 🎯 CRITÉRIOS DE SUCESSO GERAIS

1. **Qualidade de Código:**
   - Todos os novos módulos têm testes unitários (≥80% cobertura)
   - Todos os novos módulos têm documentação completa
   - Código segue PEP 8 e convenções do projeto
   - Code review para todos os PRs

2. **Performance:**
   - Não degradação de performance com novas features
   - Otimização de bottlenecks identificados
   - Métricas de performance monitorizadas

3. **Robustez:**
   - Zero crashes inesperados em produção
   - Graceful degradation para todos os subsistemas
   - Recovery automático para todos os erros conhecidos

4. **Observabilidade:**
   - Todas as métricas visíveis no dashboard
   - Alertas automáticos para anomalias
   - Logs estruturados e pesquisáveis

5. **Documentação:**
   - AGENTS.md atualizado com cada nova feature
   - Tutoriais para utilizadores avançados
   - Documentação de API para desenvolvedores

---

## 🚀 CONCLUSÃO

Este plano transforma o Soberana Omega de um bot funcional para um sistema profissional máximo. A abordagem é faseada, priorizando robustez antes de inteligência, e autonomia antes de automação.

**Principios:**
1. **Qualidade > Quantidade:** Melhor fazer menos features bem feitas
2. **Robustez > Features:** De nada serve ser inteligente se crasha
3. **Autonomia > Automação:** O objetivo é o bot melhorar sozinho
4. **Testes > Intuição:** Sem testes, não há confiança
5. **Documentação > Código:** Código sem documentação é inútil

**Resultado Final:**
Um bot que não só joga Brawl Stars - ele domina, evolui, e é indistinguível de um jogador humano profissional.

---

**Fim do Plano**
