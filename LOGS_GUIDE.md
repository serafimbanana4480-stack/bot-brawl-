# Guia do Sistema de Logs - Soberana Omega Bot

## Visão Geral

O Soberana Omega Bot possui um sistema de logs avançado e estruturado que permite perceber **tudo que o bot está fazendo** em tempo real. O sistema foi recentemente melhorado com novas categorias, logs estruturados com dados adicionais, e integração em todos os componentes principais.

## Arquitetura

### Componentes

**1. RealtimeLogManager** (`realtime_logs.py`)
- Gerencia logs em tempo real
- Mantém histórico de logs (configurável)
- Envia logs para WebSocket connections
- Suporta filtros por categoria e nível

**2. WebSocketLogHandler** (`realtime_logs.py`)
- Handler de logging que envia para WebSocket
- Permite ver logs em tempo real no dashboard
- Determina automaticamente a categoria do log

**3. Integração em Componentes**
- Lobby Automator - Logs de seleção de brawlers, press play, etc.
- Play Logic - Logs de combate, detecção de inimigos, etc.
- State Manager - Logs de transições de estado, timeouts, etc.
- Auto Tuner - Logs de ajustes de parâmetros, análise de performance
- Safety System - Logs de emergência, modo furtivo, etc.

## Categorias de Logs

### Categorias Atuais

**system** - Eventos gerais do sistema
- Inicialização
- Erros do sistema
- Eventos não categorizados

**vision** - Eventos de visão computacional
- Detecção de objetos
- Tracking de inimigos
- Inferência de modelos

**match** - Eventos de partidas
- Início de partida
- Fim de partida
- Resultados

**safety** - Eventos de segurança
- Ativações de segurança
- Emergency stops
- Modo furtivo
- Pausas forçadas

**control** - Eventos de controle
- Controle de emulador
- ADB commands
- Screen automation

**lobby** - Eventos do lobby (NOVO)
- Seleção de brawlers
- Press play
- Troca de brawlers
- Fechamento de popups

**combat** - Eventos de combate (NOVO)
- Ações de combate
- Detecção de inimigos
- Movimentos
- Tiros

**state** - Eventos de estado (NOVO)
- Transições de estado
- Detecção de mapa
- Timeouts
- Handlers executados

**auto_tuning** - Eventos de auto-tuning (NOVO)
- Ajustes de parâmetros
- Análise de performance
- Reset de parâmetros

**brawler** - Eventos específicos de brawlers (NOVO)
- Brawler selecionado
- Estratégias específicas

**humanization** - Eventos de humanização (NOVO)
- Ações de humanização
- Variações de delay

## Níveis de Log

**DEBUG** - Informações detalhadas para debugging
- Informações de desenvolvimento
- Detalhes de implementação
- Estados internos

**INFO** - Informações gerais
- Eventos importantes
- Mudanças de estado
- Operações bem-sucedidas

**WARNING** - Avisos
- Problemas não críticos
- Comportamentos inesperados
- Falhas temporárias

**ERROR** - Erros
- Erros de execução
- Falhas de operação
- Exceções

**CRITICAL** - Erros críticos
- Emergency stops
- Falhas de segurança
- Sistema não funcional

## Logs Estruturados

### Formato

```python
log_manager.log(
    message="Descrição do evento",
    level="INFO",
    category="lobby",
    data={
        "chave1": "valor1",
        "chave2": "valor2"
    }
)
```

### Exemplos

**Seleção de Brawler:**
```python
log_manager.log(
    message="Brawler selecionado via OCR: colt",
    level="INFO",
    category="lobby",
    data={
        "brawler_name": "colt",
        "ocr_backend": "easyocr",
        "confidence": 0.85,
        "click_x": 960,
        "click_y": 540,
        "attempt": 1
    }
)
```

**Transição de Estado:**
```python
log_manager.log(
    message="Transição de estado: lobby -> in_game",
    level="INFO",
    category="state",
    data={
        "from_state": "lobby",
        "to_state": "in_game"
    }
)
```

**Combate:**
```python
log_manager.log(
    message="Detecção de combate: 3 inimigos",
    level="INFO",
    category="combat",
    data={
        "enemies_count": 3,
        "bushes_count": 2,
        "player_position": [960, 540]
    }
)
```

**Auto-Tuning:**
```python
log_manager.log(
    message="Ciclo de auto-tuning concluído: success=True",
    level="INFO",
    category="auto_tuning",
    data={
        "success": true,
        "adjustments": {"attack_distance": 10},
        "analysis": {"win_rate": 0.55}
    }
)
```

**Emergency Stop:**
```python
log_manager.log(
    message="EMERGENCY STOP ativado: APM alto",
    level="CRITICAL",
    category="safety",
    data={
        "action": "emergency_stop",
        "reason": "APM alto"
    }
)
```

## API de Logs

### Endpoint: `/api/brawl-stars/logs`

**Método:** GET

**Parâmetros:**
- `n` (opcional, padrão: 100) - Número de logs recentes
- `category` (opcional) - Filtrar por categoria
- `level` (opcional) - Filtrar por nível
- `search` (opcional) - Buscar texto nas mensagens

**Exemplos:**

```bash
# Obter últimos 100 logs
curl http://localhost:8003/api/brawl-stars/logs

# Filtrar por categoria
curl "http://localhost:8003/api/brawl-stars/logs?category=lobby"

# Filtrar por nível
curl "http://localhost:8003/api/brawl-stars/logs?level=ERROR"

# Buscar texto
curl "http://localhost:8003/api/brawl-stars/logs?search=emergency"

# Combinar filtros
curl "http://localhost:8003/api/brawl-stars/logs?category=lobby&level=WARNING&n=50"
```

**Resposta:**
```json
{
  "success": true,
  "timestamp": "2026-05-08T00:00:00",
  "logs": [
    {
      "timestamp": "2026-05-08T00:00:00",
      "level": "INFO",
      "message": "Brawler selecionado via OCR: colt",
      "category": "lobby",
      "data": {
        "brawler_name": "colt",
        "confidence": 0.85
      }
    }
  ],
  "count": 1,
  "filters": {
    "n": 100,
    "category": "lobby",
    "level": "WARNING",
    "search": null
  },
  "stats": {
    "total_logs": 1500,
    "by_category": {
      "lobby": 300,
      "combat": 500,
      "state": 200
    },
    "by_level": {
      "INFO": 800,
      "WARNING": 200,
      "ERROR": 50
    }
  }
}
```

## Eventos Logados

### Lobby Events

**Seleção de Brawler:**
- Início da seleção
- Candidatos OCR encontrados
- Clique em brawler
- Confirmação de seleção
- Falha na seleção

**Press Play:**
- Pressionando botão Play
- Resgate de recompensas
- Sucesso/Falha

**Troca de Brawler:**
- Início da troca
- Seleção do próximo brawler
- Sucesso/Falha da troca

**Popups:**
- Detecção de popup
- Fechamento de popup

### Combat Events

**Round de Combate:**
- Início do round
- Detecção de objetos
- Jogador detectado
- Inimigos detectados
- Bushes detectados
- Inimigo mais próximo

**Ações de Combate:**
- Movimentos
- Tiros
- Coleta de power cubes
- Uso de super
- Uso de gadget

### State Events

**Transições de Estado:**
- Transição de estado (from -> to)
- Detecção de mapa
- Handler executado
- Duração de handler

**Timeouts:**
- Estado preso por timeout
- Reset para lobby
- Reset de match controller

**Handlers:**
- Handler iniciado
- Handler concluído
- Falha no handler

### Auto-Tuning Events

**Ciclo de Tuning:**
- Início do ciclo
- Análise de performance
- Ajustes calculados
- Ajustes aplicados
- Ciclo concluído

**Parâmetros:**
- Ajuste de attack_distance
- Ajuste de shot_cooldown
- Ajuste de safety_threshold
- Ajuste de aggressiveness

### Safety Events

**Emergency Stop:**
- Ativação de emergency stop
- Razão do emergency stop

**Modo Furtivo:**
- Ativação do modo furtivo
- Desativação do modo furtivo

**Pausas:**
- Pausa forçada
- Pausa obrigatória
- Retorno de pausa

## Monitoramento em Tempo Real

### WebSocket

O sistema suporta WebSocket para monitoramento em tempo real dos logs.

**Conexão:**
```javascript
const ws = new WebSocket('ws://localhost:8003/ws/logs');

ws.onmessage = (event) => {
  const log = JSON.parse(event.data);
  console.log(log);
};
```

### Dashboard

Logs podem ser visualizados em tempo real no dashboard do bot (se disponível).

## Boas Práticas

### Quando Usar Logs Estruturados

**Use logs estruturados quando:**
- O evento tem dados adicionais importantes
- Precisa filtrar/analizar logs posteriormente
- O evento é crítico para debugging
- Precisa rastrear fluxo de execução

**Exemplo:**
```python
# BOM - Log estruturado
log_manager.log(
    message="Brawler selecionado",
    level="INFO",
    category="lobby",
    data={"brawler": "colt", "confidence": 0.85}
)

# OK - Log simples
logger.info("Brawler colt selecionado")
```

### Níveis de Log

**DEBUG:**
- Informações de desenvolvimento
- Detalhes de implementação
- Estados internos

**INFO:**
- Eventos importantes
- Mudanças de estado
- Operações bem-sucedidas

**WARNING:**
- Problemas não críticos
- Comportamentos inesperados
- Falhas temporárias

**ERROR:**
- Erros de execução
- Falhas de operação
- Exceções

**CRITICAL:**
- Emergency stops
- Falhas de segurança
- Sistema não funcional

### Categorias

Use a categoria mais específica possível:
- `lobby` para eventos do lobby
- `combat` para eventos de combate
- `state` para transições de estado
- `auto_tuning` para ajustes de parâmetros
- `safety` para eventos de segurança
- `system` para eventos gerais

## Troubleshooting

### Logs Não Aparecendo

**Verificar:**
1. Log manager está inicializado?
2. WebSocket está conectado?
3. Categoria está correta?
4. Nível de log está configurado?

**Solução:**
```python
# Verificar se log manager está disponível
from brawl_bot.realtime_logs import get_log_manager
log_manager = get_log_manager()
print(log_manager.get_stats())
```

### Logs Não Estruturados

**Verificar:**
1. Componente integra log_manager?
2. Log estruturado está sendo chamado?
3. Data field está sendo preenchido?

**Solução:**
```python
# Adicionar log estruturado
if log_manager:
    log_manager.log(
        message="Evento",
        level="INFO",
        category="lobby",
        data={"key": "value"}
    )
```

### Filtros Não Funcionando

**Verificar:**
1. Nome da categoria está correto?
2. Nível está em maiúsculas?
3. Search está em minúsculas?

**Solução:**
```bash
# Usar categoria correta
curl "http://localhost:8003/api/brawl-stars/logs?category=lobby"

# Usar nível em maiúsculas
curl "http://localhost:8003/api/brawl-stars/logs?level=ERROR"

# Search é case-insensitive
curl "http://localhost:8003/api/brawl-stars/logs?search=emergency"
```

## Resumo das Melhorias

### Antes
- Categorias genéricas (system, vision, match, safety, control)
- Logs não estruturados
- `log_manager` usado apenas em 2 lugares
- Eventos importantes não logados

### Depois
- 11 categorias específicas (lobby, combat, state, auto_tuning, brawler, humanization)
- Logs estruturados com dados adicionais
- `log_manager` integrado em todos os componentes principais
- Todos os eventos importantes logados
- API melhorada com filtros e busca
- Estatísticas de logs

## Conclusão

O sistema de logs melhorado permite **perceber tudo que o bot está fazendo** com:
- Categorias específicas para cada tipo de evento
- Logs estruturados com dados adicionais
- Integração em todos os componentes principais
- API robusta com filtros e busca
- Monitoramento em tempo real via WebSocket

Para mais informações, consulte:
- `realtime_logs.py` - Implementação do sistema de logs
- `api.py` - Endpoint de logs
- Componentes individuais para exemplos de uso
