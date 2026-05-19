# Diagnóstico do Bot - Descobertas

## Problema Principal: Brawl Stars Não Está Aberto no Emulador

O diagnóstico visual revelou que o bot está capturando screenshots do **desktop/Windows**, não do Brawl Stars. A análise das screenshots mostra:

- **Centro escuro** (mean=7.2) - não é a tela do jogo
- **0 pixels amarelos** na região do botão Play - o botão não existe na screenshot
- **Estado detectado: `unknown`** (confiança 0.00) - o detector não reconhece a tela

### Verificação no Setup (Nova)
Adicionei verificação no `wrapper.py` que agora alerta se o Brawl Stars não está aberto:
```
[WRAPPER] Brawl Stars NÃO detectado no emulador! Verifique se o jogo está aberto.
```

## O Que Acontece Quando o Jogo Não Está Aberto

1. Bot detecta o emulador (BlueStacks) ✓
2. Bot captura screenshot do emulador ✓
3. Mas o screenshot mostra o desktop, não o jogo ✗
4. `UnifiedStateDetector` retorna `unknown` para todos os estados ✗
5. `SmartPlayButtonDetector` não encontra o botão Play ✗
6. Bot fica preso em loops de tentativa ❌

## Solução Imediata

**Antes de iniciar o bot, certifique-se de:**
1. ✅ BlueStacks está aberto
2. ✅ Brawl Stars está aberto DENTRO do BlueStacks
3. ✅ Brawl Stars está na tela inicial/lobby (não minimizado)
4. ✅ A janela do BlueStacks está visível (não minimizada)

## Melhorias no Código (Já Aplicadas)

1. **Logging detalhado no `lobby_navigator.py`** - agora mostra se o botão Play foi encontrado e onde
2. **Threshold ajustado** - de 0.45 para 0.30 para ser mais permissivo na detecção
3. **Verificação no setup** - o bot agora detecta se o jogo não está aberto e alerta
4. **Correção do JS do dashboard** - `\n\n` escapado corretamente

## Próximos Passos para Melhorar "Noção do Jogo"

1. **Adicionar detecção de estado por cor dominante** - identificar se estamos em lobby (amarelo), loading (verde), ou in-game (escuro)
2. **Melhorar o fallback do detector** - quando pixel/template falham, usar OCR ou análise de regiões
3. **Adicionar timeout inteligente** para matchmaking - se demorar mais que 20s, tentar clicar novamente ou verificar conexão
4. **Logs de visão** - mostrar quantos objetos o YOLO detecta por frame (inimigos, power-ups, etc.)
