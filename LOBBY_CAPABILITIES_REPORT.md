# Relatório de Capacidades do Bot - Lobby e Percepção do Jogo

## Resumo Executivo

O Soberana Omega Bot possui **capacidades avançadas de percepção do jogo e automação do lobby**, incluindo seleção inteligente de brawlers via OCR, detecção de estados do jogo, e automação completa do fluxo do lobby.

## 1. Percepção do Jogo (State Finder)

### Arquivo: `pylaai_real/state_finder.py`

### Capacidades de Detecção de Estado

**Tecnologia:** Template Matching com OpenCV

**Estados Detectados:**
- ✅ **Lobby** - Tela inicial do jogo
- ✅ **Matchmaking** - Procurando partida
- ✅ **Loading** - Carregando partida
- ✅ **Connection Lost** - Perda de conexão
- ✅ **End** - Tela de fim de partida
- ✅ **In Game** - Durante a partida

**Funcionalidades:**
- **Identificação de Mapas:** Detecta nome do mapa a partir de keywords no hint
  - Island Invasion, Canyon Crossing, Brawl Ball, Gem Grab, Bounty, Heist, Showdown, Solo Showdown, etc.
- **Mapeamento de Hints:** Converte hints da automação de tela para estados explícitos
- **Diagnóstico:** Guarda último diagnóstico para debugging
- **Template Cache:** Cache de templates para performance

**Limitações:**
- Dependente de templates em `lobby.toml`
- Requer imagens de referência para cada estado
- Pode ter falsos positivos em telas similares

## 2. Automação do Lobby (Lobby Automator)

### Arquivo: `pylaai_real/lobby_automator.py`

### Capacidades Completas

#### 2.1 Seleção de Brawlers com OCR

**Tecnologia:** OCR (EasyOCR, Tesseract, ou PaddleOCR)

**Funcionalidades:**
- ✅ **Seleção por Nome:** Seleciona brawler específico pelo nome
- ✅ **OCR Multi-Backend:** Suporta EasyOCR, Tesseract, PaddleOCR
- ✅ **Scroll Inteligente:** Navega pela lista/grid de brawlers
- ✅ **Confirmação de Seleção:** Confirma se o brawler foi selecionado
- ✅ **Detecção de Layout:** Identifica se é lista ou grid
- ✅ **Preprocessamento:** Reduz imagem para melhor performance do OCR
- ✅ **Confidence Threshold:** Usa threshold de 0.6 para match

**Processo de Seleção:**
1. Captura screenshot
2. Reduz imagem para 65% do tamanho original
3. Executa OCR para detectar textos
4. Busca nome do brawler nos resultados
5. Clica na posição do texto encontrado
6. Confirma seleção
7. Define brawler no play_logic para estratégias específicas

**Limitações:**
- Requer OCR instalado (EasyOCR, Tesseract ou PaddleOCR)
- Pode falhar se OCR não estiver disponível
- Requer brawler visível na tela (scroll necessário)
- Pode ter erros em nomes similares

#### 2.2 Fila de Brawlers (Brawler Queue)

**Funcionalidades:**
- ✅ **Fila Prioritária:** Ordena brawlers por prioridade (1-5)
- ✅ **Metas de Troféus:** Troca automaticamente ao atingir meta
- ✅ **Metas de Vitórias:** Troca automaticamente após X vitórias
- ✅ **Detecção de Derrotas:** Troca após 3 derrotas seguidas
- ✅ **Avanço Automático:** Avança para próximo brawler ativo
- ✅ **Peek Next:** Visualiza próximo sem alterar índice
- ✅ **Reordenação:** Reordena fila manualmente

**Configuração por Brawler:**
- `name`: Nome do brawler
- `current_trophies`: Troféus atuais
- `target_trophies`: Meta de troféus
- `current_wins`: Vitórias atuais
- `target_wins`: Meta de vitórias
- `priority`: Prioridade (1-5)
- `enabled`: Se está ativo

#### 2.3 Funções de Lobby

**Press Play:**
- ✅ **Clicar no Botão Play:** Clica em coordenadas (960, 950)
- ✅ **Resgate de Recompensas:** Clica no centro (960, 540) para limpar recompensas
- ✅ **Fallback:** Clica mais abaixo (960, 1000) como backup
- ✅ **Integração Screen Automation:** Coordena com automação de tela

**Close Popup:**
- ✅ **Fechar Popups Inteligentemente:** Fecha popups de recompensas, amigos, ofertas
- ✅ **Clicar no "X":** Clica no canto superior direito (1800, 100)
- ✅ **Clicar no Centro:** Para recompensas de "Toque para abrir"
- ✅ **ESC como Último Recurso:** Pressiona ESC se não funcionar

**Check and Switch:**
- ✅ **Verificar Metas:** Verifica se brawler atingiu metas
- ✅ **Trocar Automaticamente:** Troca para próximo brawler se necessário
- ✅ **Avançar Fila:** Avança para próximo brawler na fila

#### 2.4 Diagnóstico e Logging

**Funcionalidades:**
- ✅ **Diagnostic Report:** Relatório detalhado de cada operação
- ✅ **Step-by-Step Logging:** Log de cada passo da seleção
- ✅ **OCR Results:** Log de resultados do OCR
- ✅ **Click Coordinates:** Log de coordenadas clicadas
- ✅ **Error Tracking:** Rastreamento de erros

**Informações Diagnosticadas:**
- Estado atual
- Razão do estado
- Tentativas de scroll
- Backend OCR usado
- Candidatos encontrados
- Confiança dos matches
- Posições clicadas
- Confirmação de seleção

## 3. Integração com Outros Componentes

### 3.1 Play Logic
- **set_current_brawler:** Define brawler atual para estratégias específicas
- **Brawler-Specific Strategies:** Usa estratégias diferentes por brawler

### 3.2 Screen Automation
- **Coordenação de Cliques:** Coordena cliques manuais com automação de tela
- **Pausa Temporária:** Pausa automação durante cliques manuais

### 3.3 Emulator Controller
- **Integração ADB:** Usa ADB para cliques se disponível
- **Fallback pyautogui:** Usa pyautogui se ADB não disponível

### 3.4 Progress Observer
- **Verificação de Progresso:** Verifica se deve trocar brawler
- **Detecção de Fim de Partida:** Detecta fim de partida para trocar

## 4. Limitações e Dependências

### Dependências Obrigatórias
- ✅ OpenCV (template matching)
- ✅ PIL/Pillow (processamento de imagem)
- ✅ pyautogui (cliques manuais)
- ✅ numpy (processamento de arrays)

### Dependências Opcionais
- ⚠️ EasyOCR (OCR principal)
- ⚠️ Tesseract (OCR alternativo)
- ⚠️ PaddleOCR (OCR alternativo)
- ⚠️ ADB (cliques via ADB)

### Limitações Conhecidas
1. **OCR Dependence:** Seleção de brawlers requer OCR instalado
2. **Template Matching:** Depende de templates em `lobby.toml`
3. **Coordenadas Fixas:** Alguns cliques usam coordenadas fixas (1920x1080)
4. **Scroll Limitado:** Scroll pode não funcionar em todos os layouts
5. **Nomes de Brawlers:** OCR pode falhar com nomes similares ou não padrão
6. **Performance:** OCR pode ser lento em máquinas mais antigas

## 5. Capacidades Atuais vs Futuras

### ✅ Capacidades Atuais (Implementadas)
- Detecção de estado do jogo via template matching
- Seleção de brawlers via OCR
- Fila de brawlers com metas
- Troca automática baseada em performance
- Press play com resgate de recompensas
- Fechamento inteligente de popups
- Diagnóstico detalhado
- Integração com play logic
- Coordenação com screen automation

### 🚧 Capacidades Futuras (Sugestões)
- Detecção de brawlers via visão computacional (YOLO)
- Seleção de brawlers por imagem (não apenas nome)
- Detecção automática de coordenadas (adaptar a diferentes resoluções)
- Detecção de eventos especiais no lobby
- Seleção de modos de jogo
- Detecção de disponibilidade de brawlers (bloqueados/não desbloqueados)
- Melhoria do OCR com fine-tuning
- Detecção de brawlers raros/estrelados

## 6. Conclusão

O Soberana Omega Bot possui **capacidades robustas de percepção do jogo e automação do lobby**:

**Pontos Fortes:**
- ✅ Detecção de estado do jogo funcional
- ✅ Seleção de brawlers via OCR inteligente
- ✅ Fila de brawlers com metas automáticas
- ✅ Automação completa do fluxo do lobby
- ✅ Diagnóstico detalhado para debugging
- ✅ Integração com múltiplos componentes

**Pontos de Melhoria:**
- ⚠️ Dependência de OCR para seleção de brawlers
- ⚠️ Coordenadas fixas para alguns cliques
- ⚠️ Pode falhar em layouts não padrão
- ⚠️ Performance do OCR pode ser lenta

**Recomendações:**
1. Garantir que OCR esteja instalado (EasyOCR recomendado)
2. Verificar se `lobby.toml` tem templates corretos
3. Testar seleção de brawlers antes de usar em produção
4. Monitorar logs de diagnóstico para identificar problemas
5. Considerar implementar detecção de brawlers via YOLO para maior robustez

## 7. Como Testar as Capacidades

### Testar Detecção de Estado
```python
from pylaai_real.state_finder import StateFinder
from pathlib import Path

state_finder = StateFinder(Path("images"))
state = state_finder.find_state(screenshot)
print(f"Estado: {state}")
```

### Testar Seleção de Brawlers
```python
from pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig

queue = BrawlerQueue()
queue.add_brawler(BrawlerConfig(name="colt", target_trophies=400))
lobby = LobbyAutomator(queue)

# Selecionar brawler atual
success = lobby.select_current_brawler(screenshot_func)
print(f"Seleção: {success}")
```

### Testar Fila de Brawlers
```python
# Adicionar brawlers
queue.add_brawler(BrawlerConfig(name="colt", priority=1))
queue.add_brawler(BrawlerConfig(name="shelly", priority=2))

# Verificar metas
if queue.check_goals():
    next_brawler = queue.next()
    print(f"Próximo brawler: {next_brawler.name}")
```

## 8. Arquivos Relacionados

- `pylaai_real/state_finder.py` - Detecção de estado do jogo
- `pylaai_real/lobby_automator.py` - Automação do lobby
- `pylaai_real/state_manager.py` - Gerenciamento de estados
- `pylaai_real/play.py` - Lógica de jogo (usa brawler selecionado)
- `match_controller.py` - Controle de partidas (usa fila de brawlers)
- `images/lobby.toml` - Configuração de templates do lobby
