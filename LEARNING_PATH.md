# Como o Bot Aprenderia a Jogar Melhor

## Estado Atual

O bot atualmente usa **regras hard-coded**:
- Lógica de movimento baseada em distância e posição
- Aim assist com predição linear
- Brawler queue baseada em troféus e wins
- Safety system com thresholds fixos

## Componentes de Aprendizado Existentes

### 1. MatchController - Rastreamento de Performance
- **Arquivo:** `match_controller.py`
- **Função:** Rastreia kills, deaths, damage, win/loss por partida
- **Uso:** Base para métricas de performance e auto-tuning

### 2. Retrain System - Auto-Retreinamento
- **Arquivo:** `training/retrain.py`
- **Função:** Monitora performance e dispara retraining quando necessário
- **Métricas:** kills, deaths, damage_dealt, detection_accuracy, tracking_consistency, good/bad_decisions
- **Status:** Parcialmente implementado (métricas definidas, loop de treinamento não)

### 3. Behavior Cloning - Imitação de Jogadores Humanos
- **Arquivo:** `rl_stubs/behavior_cloning.py`
- **Função:** Treina política imitando ações de jogadores humanos
- **Requer:** Dataset de (frame, action) pairs anotados
- **Status:** STUB (interface definida, treinamento não implementado)

### 4. CQL Trainer - Offline Reinforcement Learning
- **Arquivo:** `rl_stubs/cql_trainer.py`
- **Função:** Conservative Q-Learning para aprendizado offline
- **Requer:** Replay buffer de transições (state, action, reward, next_state, done)
- **Status:** STUB (interface definida, treinamento não implementado)

---

## Como o Bot Aprenderia (3 Abordagens)

### Abordagem 1: Auto-Tuning de Parâmetros (Mais Fácil)

**Como funciona:**
- Analisa histórico de partidas (match_controller)
- Ajusta parâmetros baseados em performance
- Exemplos de parâmetros ajustáveis:
  - Distância ideal de ataque
  - Thresholds de segurança
  - Cooldowns de habilidades
  - Padrões de movimento

**Implementação:**
```python
# Exemplo: Ajustar distância de ataque baseada em win rate
if match_controller.win_rate < 0.5:
    # Aumentar distância de ataque
    play_logic.attack_distance += 50
elif match_controller.win_rate > 0.7:
    # Diminuir distância de ataque (mais agressivo)
    play_logic.attack_distance -= 20
```

**Benefícios:**
- Fácil de implementar
- Não requer dataset
- Melhorias imediatas

**Limitações:**
- Apenas ajusta parâmetros existentes
- Não aprende novas estratégias

---

### Abordagem 2: Behavior Cloning (Média Dificuldade)

**Como funciona:**
- Grava gameplay de jogadores humanos ou do próprio bot
- Treina rede neural para imitar ações
- A rede aprende: frame → ação (move, attack, target)

**Dataset necessário:**
```json
{
  "frame_path": "screenshot_001.png",
  "action": {
    "move_angle": 45.0,
    "attack": true,
    "use_super": false,
    "target_x": 0.5,
    "target_y": 0.3
  },
  "game_state": "combat"
}
```

**Implementação:**
1. Gravar gameplay com `screenshot_recorder.py`
2. Anotar ações tomadas (ou usar ações do bot)
3. Treinar CNN+MLP para prever ações
4. Substituir lógica hard-coded pela rede treinada

**Benefícios:**
- Aprende estratégias complexas
- Imita comportamento humano
- Generaliza para situações similares

**Limitações:**
- Requer dataset de qualidade
- Não melhora além do dataset
- Pode copiar erros humanos

---

### Abordagem 3: Offline RL (Alta Dificuldade)

**Como funciona:**
- Coleta replay buffer de transições
- Usa CQL (Conservative Q-Learning) para treinar offline
- Aprende política ótima baseada em recompensas

**Replay buffer necessário:**
```json
{
  "state": "frame_embedding",
  "action": 5,
  "reward": 1.0,
  "next_state": "next_frame_embedding",
  "done": false,
  "game_state": "combat"
}
```

**Função de recompensa:**
```python
def calculate_reward(old_state, action, new_state):
    reward = 0
    
    # Recompensa por dano causado
    reward += (new_state.damage_dealt - old_state.damage_dealt) * 0.1
    
    # Penalidade por dano recebido
    reward -= (new_state.damage_taken - old_state.damage_taken) * 0.2
    
    # Recompensa por kill
    if new_state.kills > old_state.kills:
        reward += 10
    
    # Penalidade por morte
    if new_state.health <= 0:
        reward -= 20
    
    # Recompensa por vitória
    if new_state.match_result == "win":
        reward += 50
    
    return reward
```

**Implementação:**
1. Gravar transições durante gameplay
2. Calcular recompensas baseadas em métricas
3. Treinar CQL offline
4. Deploy da política treinada

**Benefícios:**
- Aprende estratégias ótimas
- Melhora além do dataset inicial
- Adaptativo a mudanças no jogo

**Limitações:**
- Requer replay buffer grande
- Complexo de implementar
- Risco de overfitting

---

## Caminho Recomendado

### Fase 1: Auto-Tuning (Semanal)
1. Implementar auto-tuning de parâmetros baseado em match history
2. Ajustar distância de ataque, thresholds de segurança
3. Ajustar estratégias por mapa baseadas em win rate

### Fase 2: Behavior Cloning (Mensal)
1. Gravar 100+ partidas do bot jogando
2. Treinar policy de behavior cloning
3. A/B test entre regras hard-coded e policy treinada
4. Iterar com mais dados

### Fase 3: Offline RL (Trimestral)
1. Implementar sistema de recompensas
2. Coletar replay buffer de 1000+ partidas
3. Treinar CQL offline
4. Deploy e monitorar performance

---

## Próximos Passos Imediatos

### 1. Habilitar Coleta de Dados
```python
# Adicionar em play.py
def record_transition(self, state, action, reward, next_state):
    """Grava transição para replay buffer"""
    transition = {
        "state": self._encode_state(state),
        "action": action,
        "reward": reward,
        "next_state": self._encode_state(next_state),
        "done": self._is_match_over(),
        "game_state": self.current_game_state
    }
    self.replay_buffer.append(transition)
```

### 2. Implementar Função de Recompensa
```python
def calculate_reward(self, old_snapshot, new_snapshot):
    reward = 0
    
    # Dano causado
    reward += (new_snapshot.damage_dealt - old_snapshot.damage_dealt) * 0.1
    
    # Dano recebido
    reward -= (new_snapshot.damage_taken - old_snapshot.damage_taken) * 0.2
    
    # Kills
    reward += (new_snapshot.kills - old_snapshot.kills) * 10
    
    # Morte
    if new_snapshot.health <= 0:
        reward -= 20
    
    # Vitória
    if new_snapshot.match_result == "win":
        reward += 50
    
    return reward
```

### 3. Auto-Tuning Simples
```python
def auto_tune_parameters(self):
    """Ajusta parâmetros baseado em performance recente"""
    history = self.match_controller.get_stats(last_n=50)
    
    if history["win_rate"] < 0.4:
        # Mais conservador
        self.attack_distance += 50
        self.safety_threshold += 0.1
    elif history["win_rate"] > 0.6:
        # Mais agressivo
        self.attack_distance -= 20
        self.safety_threshold -= 0.05
```

---

## Conclusão

O bot tem a **infraestrutura** para aprender (match_controller, retrain system, RL stubs), mas precisa de:

1. **Implementação dos loops de treinamento** (behavior_cloning, CQL)
2. **Coleta de dados** (transições, replay buffer)
3. **Função de recompensa** bem definida
4. **Iteração contínua** (treinar, deploy, monitorar)

**Recomendação:** Começar com auto-tuning de parâmetros (fácil, rápido benefício), depois behavior cloning (média), e finalmente offline RL (avançado).
