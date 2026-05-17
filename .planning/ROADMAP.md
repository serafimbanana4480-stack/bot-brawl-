# ROADMAP - Milestone v2.0: OCR Avançado e Visão Computacional

## Visão Geral

Este roadmap cobre a evolução do sistema de visão do bot Brawl Stars, de heurísticas pixel + OCR básico para um pipeline multimodal robusto que extrai informações completas do HUD em tempo real.

---

## Fase 1: OCR Avançado do HUD

**Status:** COMPLETE ✅
**Descrição:** Criar sistema de extração de valores numéricos do HUD com pré-processamento, múltiplas variantes e fallback.
**Ficheiros Criados:** `vision/ocr_hud_extractor.py`
**Ficheiros Modificados:** `tests/test_ocr_hud_extractor.py`

**Tarefas:**
- [x] Criar OCRHudExtractor com suporte a HP, ammo, super, timer, score
- [x] Implementar pré-processamento robusto (scale, threshold, denoise)
- [x] Adicionar múltiplas ROIs por campo com votação
- [x] Implementar normalização de texto numérico (O→0, I→1, etc.)
- [x] Criar fallback para heurísticas de pixel quando OCR falha
- [x] Adicionar cache de resultados com TTL adaptativo
- [x] Testes unitários com mock de screenshots

**Critérios de Sucesso:**
- [x] Precisão ≥ 90% em valores numéricos em screenshots de teste
- [x] Latência < 100ms para extração completa do HUD
- [x] Testes passam (50/50)

---

## Fase 2: Detecção de Estado do Jogador

**Status:** COMPLETE ✅
**Descrição:** Combinar YOLO + OCR + heurísticas para determinar estado completo do jogador.
**Ficheiros Criados:** `vision/player_state_detector.py`
**Ficheiros Modificados:** `tests/test_player_state_detector.py`

**Tarefas:**
- [x] Criar PlayerState dataclass (vivo/morto, super, gadget, arbusto, perigo)
- [x] Implementar fusão de fontes com pesos configuráveis
- [x] Adicionar suavização temporal de estados
- [x] Detectar transições de estado (eventos)
- [x] Integrar com PlayLogic para decisões contextualizadas
- [x] Testes de integração com subsistemas existentes

**Critérios de Sucesso:**
- [x] Estados detectados com confiança ≥ 85%
- [x] Transições detectadas em < 200ms
- [x] Zero falsos positivos em testes de lobby/estados estáticos
- [x] Testes passam (38/38)

---

## Fase 3: Visão Multimodal Unificada

**Status:** COMPLETE ✅
**Descrição:** Criar pipeline unificado que funde todas as fontes de visão numa GameState coesa.
**Ficheiros Criados:** `vision/multimodal_pipeline.py`, `vision/game_state.py`
**Ficheiros Modificados:** `tests/test_multimodal_pipeline.py`

**Tarefas:**
- [x] Definir GameState dataclass unificado (objetos + texto + pixel + estado)
- [x] Criar MultimodalPipeline com execução das 3 camadas
- [x] Implementar lógica de fusão com resolução de conflitos
- [x] Adicionar métricas de qualidade por camada
- [x] Testes de performance (latência por frame)

**Critérios de Sucesso:**
- [x] Latência total < 50ms por frame em hardware de referência
- [x] GameState completo disponível para todos os subsistemas
- [x] Testes passam (30/30)

---

## Fase 4: Dataset Enriquecido e Integração

**Status:** COMPLETE ✅
**Descrição:** Integrar dados do OCR no dataset de gameplay para treinamento de modelos mais inteligentes.
**Ficheiros Criados:** `dataset/enriched_collector.py`
**Ficheiros Modificados:** `tests/test_enriched_collector.py`

**Tarefas:**
- [x] Criar EnrichedDatasetCollector que grava estado OCR com cada frame
- [x] Adicionar campos de estado do jogador ao replay buffer
- [x] Criar eventos de transição de estado (ex: "morreu", "super pronta")
- [x] Integrar com pipeline de treino contínuo (dados enriquecidos)
- [x] Validar qualidade do dataset (coverage de campos OCR)
- [x] Testes de integração end-to-end

**Critérios de Sucesso:**
- [x] Dataset contém estado completo do jogador em ≥ 95% dos frames
- [x] Pipeline de treino consome dados enriquecidos sem erros
- [x] Testes passam (12/12)

---

## Fase 5: Validação e Documentação

**Status:** COMPLETE ✅
**Descrição:** Testes completos, benchmark de performance e documentação técnica.
**Ficheiros Modificados:** `AGENTS.md`

**Tarefas:**
- [x] Criar suite de testes para OCRHudExtractor (50 tests)
- [x] Criar suite de testes para PlayerStateDetector (38 tests)
- [x] Criar suite de testes para MultimodalPipeline (30 tests)
- [x] Criar suite de testes para EnrichedCollector (12 tests)
- [x] Documentar arquitetura de visão em AGENTS.md

**Critérios de Sucesso:**
- [x] Todos os testes passam (130/130)
- [x] Documentação técnica atualizada

---

## Resumo do Milestone

| Fase | Componente Principal | Testes | Status |
|------|---------------------|--------|--------|
| 1 | vision/ocr_hud_extractor.py | 50 | COMPLETE ✅ |
| 2 | vision/player_state_detector.py | 38 | COMPLETE ✅ |
| 3 | vision/multimodal_pipeline.py | 30 | COMPLETE ✅ |
| 4 | dataset/enriched_collector.py | 12 | COMPLETE ✅ |
| 5 | Documentação (AGENTS.md) | - | COMPLETE ✅ |

**Total de testes:** 130 passing

## Próximos Passos Recomendados (pós-v2.0)
1. **v2.1 - Pipeline de Treino Avançado:** Usar dados enriquecidos para treinar BC/CQL com estado completo
2. **v2.2 - Transfer Learning:** Treinar modelos especializados por mapa/modo de jogo
3. **v2.3 - Intenção Inimiga:** Detectar padrões de movimento e prever ações inimigas

## Notas Técnicas
- Manter compatibilidade com sistema anti-ban (não aumentar APM)
- ROIs normalizadas (0-1) para suportar múltiplas resoluções
- Cache de OCR para não repetir inferência em frames similares
- Fallback sempre disponível (heurísticas pixel como último recurso)
- EasyOCR lazy-loaded (demora ~16s para importar)
