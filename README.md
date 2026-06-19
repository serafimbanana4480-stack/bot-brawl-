# ⚔️ Soberana Omega — Brawl Stars Autonomous Bot

> **Bot autónomo de Brawl Stars com visão computacional, combate adaptativo e humanização anti-ban.**

<p align="center">
  <img src="https://img.shields.io/badge/status-Active%20Development-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/YOLO-v8%2Fv11-00FFFF?logo=yolo" alt="YOLO">
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/license-MIT-lightgrey" alt="License">
  <img src="https://img.shields.io/badge/platform-Windows%2010%2F11-0078d4?logo=windows" alt="Platform">
  <img src="https://img.shields.io/badge/GPU-NVIDIA%20CUDA-76b900?logo=nvidia" alt="GPU">
</p>

---

## 🎯 O que é?

**Soberana Omega** é um agente autónomo que joga Brawl Stars por si. Combina **visão computacional em tempo real** (YOLO + ByteTrack), **tomada de decisão adaptativa** (utility AI + reinforcement learning) e **humanização comportamental** para evitar detecção anti-cheat.

O objetivo: **subir troféus eficientemente, com padrões indistinguíveis de um humano** — incluindo timing, micro-movimentos e comportamento contextual baseado no estado do jogo.

---

## ⚡ Quick Start

```bash
# 1. Clonar
git clone https://github.com/serafimbanana4480-stack/bot-brawl-.git
cd bot-brawl-

# 2. Ambiente virtual
python -m venv .venv
.venv\Scripts\activate            # Windows
# source .venv/bin/activate       # Linux/Mac

# 3. Dependências
pip install -r requirements.txt

# 4. Modelo YOLO (já vem treinado em models/yolo/brawlstars_gpu_v8s)

# 5. Correr
python main.py                    # Iniciar o bot
```

> **Primeira execução?** Abre o emulador, faz login na conta e deixa o bot calibrar a janela (~30s).

---

## ✨ Funcionalidades

### 🧠 Inteligência & Decisão
- **🎯 Visão Computacional** — YOLOv8/v11 com TensorRT para detecção de brawlers, inimigos, projéteis e bush em tempo real (~30 FPS)
- **🎮 State Machine** — Rastreia estado do jogo (lobby, match, loja, death screen) com transições suaves
- **⚖️ Utility AI** — Sistema de pontuação que escolhe a melhor ação baseado em contexto (vida, posição, super disponível)
- **🧬 Reinforcement Learning** — Curricula treinados em simulação para melhorar jogadas específicas (escapar, snipe, controle de bush)
- **🗺️ Curriculum Learning** — Progressão de dificuldade: tutorial → easy → ranked → competitive

### 🛡️ Anti-Ban & Segurança
- **🖐️ Humanização** — Curvas Bezier para movimento de joystick, variação de tempo entre ações, fadiga simulada
- **🎭 Perfil Comportamental** — Aprende o estilo de jogo preferido (agressivo, defensivo, rotativo) e adapta
- **⏰ Agendamento Inteligente** — Não joga em horas aleatórias, respeita padrões circadianos
- **📊 Telemetria Invisível** — Análise estatística para garantir que o jogo parece humano

### 🔌 Integração & API
- **🌐 FastAPI + WebSocket** — Dashboard em tempo real (kills, deaths, troféus, telemetria)
- **📡 REST API** — Controla o bot remotamente, obtém estatísticas, envia comandos
- **🔌 Sistema de Plugins** — Adiciona novos comportamentos sem tocar no core
- **📦 Ports & Adapters (Hexagonal)** — Troca BlueStacks por LDPlayer, ADB por Win32, sem mudar lógica

### 🎯 Combate
- **🎯 Aim Assist Adaptativo** — Mira segue movimento do inimigo, compensa latência
- **💨 Dodge Inteligente** — Esquiva de projéteis baseada em trajetória predita
- **🛡️ Auto-Use Super** — Ativa super no momento ótimo baseado em análise de risco
- **🌿 Controle de Bush** — Entra/sai de bushes taticamente

---

## 🏗️ Arquitetura

```
┌─────────────────────────────────────────────────────────────┐
│                    wrapper.py (entry point)                  │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
   │  Safety  │    │  Emulator   │    │   Vision    │
   │  Module  │    │   Adapter   │    │  (YOLO+TRT)│
   └────┬─────┘    └──────┬──────┘    └──────┬──────┘
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
              ┌────────────▼────────────┐
              │   core/orchestrator     │
              │      (Hexagonal)        │
              └────────────┬────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
   ┌────▼─────┐    ┌──────▼──────┐    ┌──────▼──────┐
   │ Decision │    │   Neural    │    │    Input    │
   │ (Utility │◀──▶│  (RL/TR)    │    │ (ADB/Win32)│
   │   AI)    │    └─────────────┘    └─────────────┘
   └──────────┘
```

**Princípios chave:**
- **Ports & Adapters** — fácil trocar implementações (emulador, modelo, transport)
- **Event Bus** — comunicação assíncrona entre subsistemas
- **Pipeline Stage** — processamento faseado (percepção → decisão → ação)
- **Stateful agents** — cada subsistema mantém estado e reage a eventos

---

## 📂 Estrutura do Projeto

```
bot-brawl/
├── src/                    # Código fonte principal
│   ├── core/               # Orquestração, ports, adapters
│   ├── vision/             # YOLO, ByteTrack, OCR, GameState
│   ├── decision/           # State machine, utility AI, RL
│   ├── neural/             # Transfer learning, curriculum
│   ├── api/                # FastAPI + WebSocket
│   ├── analysis/           # Performance, replay analyzers
│   ├── data/               # Coletores de dataset
│   ├── plugins/            # Sistema de plugins
│   └── utils/              # Utilitários
├── models/yolo/            # Modelos YOLO treinados
│   └── brawlstars_gpu_v8s/ # Modelo principal (GPU)
├── dataset/                # Datasets de treinamento
├── tests/                  # Testes unitários e integração
├── docs/                   # Documentação técnica
├── config.json             # Configuração principal
├── main.py                 # Entry point
└── requirements.txt        # Dependências Python
```

---

## ⚙️ Comandos

| Comando | Descrição |
|---|---|
| `python main.py` | Iniciar o bot em modo jogo |
| `python main.py --diagnostic` | Modo diagnóstico (logs verbosos, sem jogar) |
| `python main.py --learning` | Modo aprendizado (coleta dados para RL) |
| `python main.py --headless` | Correr sem interface (servidor) |
| `python api_server.py` | Iniciar API + dashboard |
| `python train_yolo.py` | Treinar/atualizar modelo YOLO |
| `python run_tests.py` | Correr suite de testes |
| `python replay_analyzer.py <file>` | Analisar gravação de jogo |

---

## 🔧 Configuração

`config.json` controla todos os aspectos:

```json
{
  "emulator": {
    "type": "bluestacks",        // bluestacks | ldplayer | memu
    "adb_port": 5555,
    "resolution": [1920, 1080]
  },
  "vision": {
    "model_path": "models/yolo/brawlstars_gpu_v8s/best.pt",
    "confidence": 0.55,
    "use_tensorrt": true
  },
  "combat": {
    "aggressiveness": 0.7,        // 0-1
    "use_super": "optimal",       // never | optimal | always
    "dodge_probability": 0.85
  },
  "safety": {
    "max_trophies": 30000,        // parar acima disto
    "session_minutes": 90,        // duração da sessão
    "humanization_level": "high", // low | medium | high
    "fingerprint_spoofing": true
  }
}
```

---

## 🧪 Testes

```bash
pytest tests/                       # Suite completa
pytest tests/unit/                  # Apenas unit tests
pytest tests/integration/           # Integration tests
pytest --cov=src --cov-report=html  # Com coverage
```

Abre `htmlcov/index.html` para ver relatório detalhado.

---

## 📚 Documentação

| Documento | Descrição |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Arquitetura detalhada (Ports & Adapters) |
| [docs/INSTALLATION_GUIDE.md](docs/INSTALLATION_GUIDE.md) | Instalação completa, drivers, emulador |
| [docs/API_DOCUMENTATION.md](docs/API_DOCUMENTATION.md) | API REST + WebSocket |
| [docs/TESTING_GUIDE.md](docs/TESTING_GUIDE.md) | Como testar o bot |
| [docs/HUMANIZATION.md](docs/HUMANIZATION.md) | Como funciona o anti-ban |
| [CHANGELOG.md](CHANGELOG.md) | Histórico de versões |

---

## ⚠️ Disclaimer

> **Este projeto é apenas para fins educacionais e de pesquisa.**
> O uso de bots em jogos online pode violar os Termos de Serviço e resultar em banimento.
> O autor não se responsabiliza pelo uso indevido desta ferramenta.
> Use por sua conta e risco.

---

## 🤝 Contribuir

Contribuições são bem-vindas! Por favor:

1. Fork o repositório
2. Cria uma branch (`git checkout -b feature/MinhaFeature`)
3. Commit as tuas mudanças (`git commit -m 'feat: adicionar MinhaFeature'`)
4. Push para a branch (`git push origin feature/MinhaFeature`)
5. Abre um Pull Request

Lê [CONTRIBUTING.md](CONTRIBUTING.md) para mais detalhes.

---

## 📄 Licença

Este projeto está licenciado sob a **MIT License** — vê [LICENSE](LICENSE) para detalhes.

---

## 🌟 Badges & Métricas

<p align="center">
  <img src="https://img.shields.io/github/stars/serafimbanana4480-stack/bot-brawl-?style=social" alt="Stars">
  <img src="https://img.shields.io/github/forks/serafimbanana4480-stack/bot-brawl-?style=social" alt="Forks">
  <img src="https://img.shields.io/github/issues/serafimbanana4480-stack/bot-brawl-" alt="Issues">
  <img src="https://img.shields.io/github/last-commit/serafimbanana4480-stack/bot-brawl-" alt="Last Commit">
</p>

---

## 🔗 Links Úteis

- 🎮 **Brawl Stars** — https://brawlstars.com
- 🤖 **YOLO** — https://github.com/ultralytics/ultralytics
- 🧠 **PyTorch** — https://pytorch.org
- 🚀 **FastAPI** — https://fastapi.tiangolo.com
- 📘 **Documentação completa** — [docs/](docs/)

---

<p align="center">
  Feito com 🦾 por <a href="https://github.com/serafimbanana4480-stack">Soberana</a>
  &nbsp;·&nbsp;
  <a href="#top">Voltar ao topo</a>
</p>
