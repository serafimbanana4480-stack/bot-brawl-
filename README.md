# 🌵 Brawl Stars Bot - Soberana Omega Module

Este módulo integra o **PylaAI** com o ecossistema Soberana Omega, adicionando camadas de segurança, humanização e controle remoto via API.

## 🚀 Estado Atual: TOTALMENTE FUNCIONAL

O projeto foi submetido a uma auditoria profunda em 20/04/2026 e todas as falhas críticas de infraestrutura foram resolvidas.

### Principais Melhorias Implementadas:
1.  **Visão Computacional Estável**: Corrigido erro de GPU (CUDA). O bot agora roda eficientemente em CPU.
2.  **Screenshot Híbrido**: Implementada captura ultrarrápida via Win32 API com fallback automático para ADB.
3.  **Templates Recuperados**: Os botões de controle (`play`, `lobby`, `end_match`) foram extraídos e estão operacionais.
4.  **Integração de Estados**: O gestor de estados agora está conectado à lógica de jogo, permitindo que o bot execute ataques e movimentos.

## 📋 Requisitos para Execução

1.  **Emulador**: BlueStacks ou LDPlayer rodando em **1920x1080**.
2.  **ADB Habilitado**: Certifique-se de que a "Depuração Android" está ligada no emulador.
3.  **Brawl Stars**: Deve estar aberto no lobby principal.

## 🛠️ Como Iniciar

### 1. Iniciar o Servidor API
```bash
cd backend
python -m brawl_bot.main --api
```

### 2. Configurar e Iniciar via API (ou Dashboard)
O bot pode ser controlado via endpoints:
-   **Setup**: `POST /api/brawl-stars/setup` (Conecta ao emulador e carrega modelos)
-   **Add Brawler**: `POST /api/brawl-stars/brawler/add` (Define qual brawler jogar)
-   **Start**: `POST /api/brawl-stars/start` (Inicia a automação)

### 3. Verificação Rápida
Para diagnosticar a instalação, use:
```bash
python -m brawl_bot.main --check
```

## 🧠 Nota sobre Modelos de IA
O bot utiliza o modelo `yolov8n.pt` como base. Para maior precisão na detecção de brawlers específicos e projéteis, recomenda-se colocar os arquivos `main_info.pt` e `brawler_id.pt` treinados na pasta `backend/brawl_bot/models/`.

## 🔒 Segurança e Humanização
-   **Curvas de Bézier**: Movimentos de mouse suaves para evitar detecção.
-   **Safety System**: Monitoramento de APM e pausas automáticas para simular comportamento humano.
-   **Randomização**: A janela do emulador é movida periodicamente para evitar fingerprints estáticos.

---

## 📐 Architecture Diagram

```mermaid
flowchart LR
    subgraph Emulator
        EM[BlueStacks/LDPlayer]
    end
    subgraph Bot
        API[API Server] -->|calls| Vision[Vision Engine]
        Vision -->|feeds| Controller[Game Controller]
        Controller -->|uses| Safety[Safety System]
        API -->|exposes| Health[/health endpoint]
    end
    EM -->|ADB/Win32| Controller
    Vision -->|TensorRT| Model[YOLO Model]
```

## 📦 Quick‑Start

![Quick‑Start GIF](images/quick_start.gif)

## 🛠️ Contributing

1. Fork the repository  
2. Create a feature branch  
3. Write tests & documentation  
4. Open a Pull Request

See `CONTRIBUTING.md` for detailed guidelines.

## 🐞 Troubleshooting

- **Emulator not detected**: Ensure ADB is in your PATH and debugging is enabled.  
- **GPU errors**: Install CUDA drivers or set `CUDA_VISIBLE_DEVICES=""` to force CPU.

---
**Desenvolvido como parte do sistema Soberana Omega**
