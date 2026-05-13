# Guia de Troubleshooting - Soberana Omega Bot

Este guia ajuda a resolver problemas comuns ao usar o bot Soberana Omega.

---

## Índice
1. [Problemas de Conexão](#problemas-de-conexão)
2. [Problemas de Emulador](#problemas-de-emulador)
3. [Problemas de Detecção](#problemas-de-detecção)
4. [Problemas de Performance](#problemas-de-performance)
5. [Problemas de Segurança](#problemas-de-segurança)
6. [Problemas de API](#problemas-de-api)
7. [Problemas de Frontend](#problemas-de-frontend)

---

## Problemas de Conexão

### Bot não conecta ao emulador

**Sintomas:**
- Erro "Failed to setup bot. Ensure emulator is running"
- Timeout ao tentar conectar ADB

**Soluções:**
1. Verifique se o emulador está rodando
2. Verifique se o ADB está instalado: `adb version`
3. Verifique se o emulador permite conexão ADB
4. Use o endpoint `/api/brawl-stars/emulators` para listar emuladores disponíveis
5. Tente conectar manualmente: `adb connect localhost:5555`

**Comando de diagnóstico:**
```bash
adb devices
```

### ADB não detecta emulador

**Sintomas:**
- `adb devices` mostra vazio
- Erro "device offline"

**Soluções:**
1. Reinicie o emulador
2. Habilite ADB nas configurações do emulador
3. Verifique se a porta ADB está correta (padrão: 5555)
4. Reinicie o servidor ADB: `adb kill-server && adb start-server`

---

## Problemas de Emulador

### Emulador não é detectado automaticamente

**Sintomas:**
- Endpoint `/api/brawl-stars/emulators` retorna lista vazia
- Bot não encontra emulador no setup

**Soluções:**
1. Verifique se o tipo de emulador é suportado (LDPlayer, BlueStacks, Nox)
2. Verifique se o emulador está minimizado ou em segundo plano
3. Use o endpoint `/api/brawl-stars/emulators` com filtro por tipo
4. Tente especificar o emulador manualmente no setup

**Exemplo de uso:**
```bash
curl "http://localhost:8000/api/brawl-stars/emulators?emulator_type=LDPlayer"
```

### Múltiplos emuladores detectados

**Sintomas:**
- Vários emuladores listados mas não sabe qual usar
- Bot conecta ao emulador errado

**Soluções:**
1. Use `/api/brawl-stars/emulators/{name}` para selecionar específico
2. Feche emuladores não utilizados
3. Configure o emulador desejado no arquivo de configuração

---

## Problemas de Detecção

### Modelo de visão não carrega

**Sintomas:**
- Erro "No vision model loaded"
- Bot não detecta objetos na tela

**Soluções:**
1. Verifique se o modelo existe no diretório `models/`
2. Baixe o modelo se necessário
3. Verifique se o formato do modelo é suportado (.onnx, .pt)
4. Verifique logs para detalhes do erro

**Endpoint de diagnóstico:**
```bash
curl http://localhost:8003/api/brawl-stars/diagnostics
```

### Detecção de objetos falha

**Sintomas:**
- Bot não detecta inimigos
- Jogador não é detectado
- Detecções incorretas

**Soluções:**
1. Verifique a qualidade da imagem do emulador
2. Ajuste o threshold de confiança do modelo
3. Verifique se o modelo está treinado para a versão atual do jogo
4. Use logs para ver detalhes das detecções: `/api/brawl-stars/logs?category=COMBAT`

### Tracker não rastreia inimigos

**Sintomas:**
- Tracker mostra 0 tracks ativos
- IDs de inimigos mudam constantemente

**Soluções:**
1. Verifique se o tracker está sendo resetado entre partidas
2. Ajuste parâmetros do tracker (max_age, min_hits)
3. Verifique logs do tracker: `/api/brawl-stars/logs?category=TRACKER`
4. Use telemetry para ver stats do tracker: `/api/brawl-stars/telemetry`

---

## Problemas de Performance

### Bot lento ou usa muita CPU

**Sintomas:**
- FPS baixo (< 20)
- CPU alta (> 80%)
- Input atrasado

**Soluções:**
1. Reduza a resolução do emulador
2. Ajuste o intervalo de captura de tela
3. Desative features não essenciais (overlay detalhado)
4. Use endpoint de performance: `/api/brawl-stars/performance`

**Ajustes recomendados:**
- Resolução: 1280x720 ou menor
- FPS do emulador: 30 ou 60
- Intervalo de captura: 50-100ms

### Bot usa muita memória

**Sintomas:**
- Memória RAM alta (> 2GB)
- Crash por falta de memória

**Soluções:**
1. Limpe logs antigos
2. Reduza o histórico de tracking
3. Ajuste o tamanho do buffer de logs
4. Use endpoint de performance para monitorar

---

## Problemas de Segurança

### Score de suspeição alto

**Sintomas:**
- Suspicion score > 50%
- Bot pausa automaticamente
- Ações não parecem naturais

**Soluções:**
1. Ajuste configurações de humanização
2. Aumente delays entre ações
3. Adicione mais variação aos movimentos
4. Verifique APM: deve estar entre 20-60

**Ajustes recomendados:**
- min_delay: 0.3-0.5s
- max_delay: 1.5-2.0s
- mistake_probability: 0.1-0.2
- tremor_amplitude: 2.0-3.0

**Endpoint de diagnóstico:**
```bash
curl http://localhost:8000/api/brawl-stars/safety-status
```

### Bot não pausa quando deveria

**Sintomas:**
- Bot não faz pausas de descanso
- Session duration excede limite
- Troféus excedem limite

**Soluções:**
1. Verifique se auto_stop_on_detection está ativado
2. Ajuste max_session_hours
3. Ajuste max_trophies
4. Verifique logs de segurança: `/api/brawl-stars/logs?category=SAFETY`

---

## Problemas de API

### Endpoint retorna erro 500

**Sintomas:**
- Erro interno do servidor
- Resposta vazia ou incompleta

**Soluções:**
1. Verifique logs do servidor
2. Verifique se o bot está inicializado
3. Tente reiniciar o servidor
4. Use endpoint de health check: `/api/brawl-stars/health`

### WebSocket não conecta

**Sintomas:**
- Logs não aparecem em tempo real
- WebSocket desconecta constantemente

**Soluções:**
1. Verifique se o endpoint está correto: `ws://localhost:8000/ws/brawl-stars`
2. Verifique se o firewall não bloqueia a conexão
3. Tente reconectar manualmente
4. Verifique logs do servidor

---

## Problemas de Frontend

### Dashboard não carrega dados

**Sintomas:**
- Status não atualiza
- Telemetry mostra "Aguardando dados"
- Erro de conexão

**Soluções:**
1. Verifique se o backend está rodando
2. Verifique se os endpoints estão acessíveis
3. Verifique console do navegador para erros
4. Tente recarregar a página

### Logs não aparecem no dashboard

**Sintomas:**
- Seção de logs vazia
- Logs não atualizam em tempo real

**Soluções:**
1. Verifique se WebSocket está conectado
2. Verifique se logs estão sendo gerados
3. Use endpoint de logs para testar: `/api/brawl-stars/logs`
4. Ajuste filtros de categoria/nível

---

## Ferramentas de Diagnóstico

### Endpoints Úteis

**Status do Bot:**
```bash
curl http://localhost:8003/api/brawl-stars/status
```

**Telemetry Completa:**
```bash
curl http://localhost:8003/api/brawl-stars/telemetry
```

**Logs Recentes:**
```bash
curl "http://localhost:8003/api/brawl-stars/logs?n=50&category=COMBAT"
```

**Diagnostics:**
```bash
curl http://localhost:8003/api/brawl-stars/diagnostics
```

**Performance:**
```bash
curl http://localhost:8003/api/brawl-stars/performance
```

**Emuladores:**
```bash
curl http://localhost:8003/api/brawl-stars/emulators
```

### Logs do Servidor

Verifique os logs do servidor para detalhes de erros:
```bash
# Se usando Uvicorn
tail -f server.log

# Se usando Docker
docker logs -f soberana-omega
```

---

## Solução de Problemas Comum

### Bot não inicia

**Checklist:**
1. Emulador está rodando? ✅
2. ADB está instalado? ✅
3. Modelo de visão existe? ✅
4. Bot foi configurado (setup)? ✅
5. Permissões de arquivo corretas? ✅

**Comandos de verificação:**
```bash
# Verificar emulador
adb devices

# Verificar modelo
ls models/

# Verificar setup
curl http://localhost:8000/api/brawl-stars/status
```

### Bot detecta mas não age

**Checklist:**
1. Jogador está sendo detectado? ✅
2. Inimigos estão sendo detectados? ✅
3. Tracker está ativo? ✅
4. Cooldown de tiro não está ativo? ✅
5. Janela do emulador está ativa? ✅

**Comandos de verificação:**
```bash
# Verificar telemetry
curl http://localhost:8003/api/brawl-stars/telemetry

# Verificar logs de combate
curl "http://localhost:8000/api/brawl-stars/logs?category=COMBAT"
```

---

## Contato e Suporte

Se o problema persistir após seguir este guia:

1. Colete logs relevantes usando os endpoints de diagnóstico
2. Documente os passos reproduzidos
3. Inclua informações do sistema:
   - Versão do Python
   - Tipo de emulador
   - Resolução do emulador
   - Versão do jogo

---

## Glossário

- **APM**: Actions Per Minute (Ações por Minuto)
- **ADB**: Android Debug Bridge
- **FPS**: Frames Per Second (Quadros por Segundo)
- **Tracker**: Sistema de rastreamento de objetos
- **Telemetry**: Sistema de monitoramento em tempo real
- **Humanization**: Sistema de humanização de comportamento
