# 🦾 Soberana — Brawl Stars Auto Bot

> **Bot autónomo para Brawl Stars no BlueStacks** — Visão computacional (YOLO), tomada de decisão adaptativa e humanização anti-ban.

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-📷-red.svg)](https://opencv.org/)
[![YOLO](https://img.shields.io/badge/YOLO-🎯-green.svg)](https://github.com/ultralytics/ultralytics)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Visão Geral

O **Soberana** é um bot autónomo que:

1. **Deteta** o ecrã do BlueStacks (ADB screenshot)
2. **Analisa** a cena com YOLO (visão computacional)
3. **Toma** decisões adaptativas (FSM — Finite State Machine)
4. **Executa** ações (joystick virtual, botões)
5. **Humaniza** o comportamento (anti-ban)
6. **Aprende** com replays (exportação para YOLO dataset)

---

## ✨ Funcionalidades Principais

| Funcionalidade | Descrição | Estado |
|----------------|-------------|--------|
| **BlueStacks Integration** | ADB screenshot + input simulation | ✅ Ativo |
| **YOLO Detection** | Deteção de inimigos, aliados, recursos | ✅ Ativo |
| **Adaptive Decision** | FSM (Finite State Machine) para comportamento | ✅ Ativo |
| **Joystick Control** | Movimento suave (anti-ban) | ✅ Ativo |
| **Anti-Ban Humanization** | Delays aleatórios, padrões humanos | ✅ Ativo |
| **Auto-PvP** | Entra em lobbies, joga partidas | ✅ Ativo |
| **Replay Recording** | Grava episódios para dataset YOLO | ✅ Ativo |
| **Dataset Export** | Exporta frames + labels (YOLO format) | ✅ Ativo |
| **YOLO Training** | Treina modelo customizado | ✅ Ativo |
| **Watchdog** | Auto-restart em caso de crash | ✅ Ativo |

---

## 🏗️ Arquitetura

```
bot brawl/
├── soberana/
│   ├── autonomous.py      # Entry point (modo autónomo)
│   ├── vision.py          # YOLO detection
│   ├── decision.py        # FSM (Finite State Machine)
│   ├── control.py         # Joystick + input simulation
│   ├── humanize.py       # Anti-ban (delays, padrões)
│   ├── replay.py         # Gravação de episódios
│   ├── export_yolo.py    # Exportação dataset YOLO
│   └── utils.py          # Utilitários (ADB, logging)
├── scripts/
│   ├── run_bot.bat       # Launcher (Windows)
│   ├── run_monitor.bat   # Watchdog (auto-restart)
│   └── train_soberana.py # Treino YOLO
├── config/
│   ├── config.yaml        # Configurações gerais
│   └── joystick.json     # Calibração do joystick
├── data/
│   ├── episodes/        # Replays gravados
│   └── datasets/        # Datasets YOLO exportados
├── models/
│   └── best.pt          # Modelo YOLO treinado
├── tests/
│   └── test_*.py        # Testes unitários
└── README.md            # Este ficheiro
```

---

## 🚀 Quick Start

### 1. Setup (Windows)

```powershell
# 1. Entrar na pasta
cd "C:\Users\rodri\Desktop\bot brawl"

# 2. Criar ambiente virtual
python -m venv .venv

# 3. Ativar ambiente
.\venv\Scripts\Activate.ps1

# 4. Instalar dependências
pip install -r requirements.txt

# 5. Instalar YOLOv8 (Ultralytics)
pip install ultralytics
```

### 2. Configurar BlueStacks

1. Abrir BlueStacks
2. Instalar Brawl Stars
3. Fazer login (conta de teste)
4. **Não** abrir o jogo ainda

### 3. Calibrar Joystick (Obrigatório!)

```powershell
# Calibração do joystick virtual
python -m soberana.autonomous --calibrate
```

Isto vai:
- Detetar a posição do joystick no ecrã
- Guardar em `config/joystick.json`
- **Fazer isto antes de jogar!**

### 4. Modos de Jogo

#### ✅ Modo Autónomo (Recomendado)

```powershell
# Launcher (com ADB check automático)
.\run_bot.bat

# Modo PvP (lobby → play → match)
.\run_bot.bat --pvpmode

# 5 partidas PvP
.\run_bot.bat --matches 5

# 5 minutos
.\run_bot.bat --duration 300

# Gravar 500 frames para dataset
.\run_bot.bat --record 500

# Sem overlay debug
.\run_bot.bat --no-overlay

# Com debug (screenshots)
.\run_bot.bat --debug
```

#### ✅ Modo Direto (Python)

```powershell
# Modo autónomo
python -m soberana.autonomous

# Modo PvP
python -m soberana.autonomous --pvpmode

# Replay de episódio
python -m soberana.autonomous --replay data/episodes/episode_001.mp4

# Exportar dataset YOLO
python -m soberana.autonomous --export-yolo data/episodes/
```

---

## 🎮 Comandos Detalhados

### ✅ Entry Points Oficiais (Recomendados)

| Comando | Descrição |
|---------|-------------|
| `run_bot.bat` | ✅ Launcher principal (ADB check automático, módulo soberana) |
| `run_bot.bat --pvpmode` | ✅ Modo PvP (lobby → play → match) |
| `run_bot.bat --matches 5` | ✅ 5 partidas PvP |
| `run_bot.bat --duration 300` | ✅ 5 minutos |
| `run_bot.bat --record 500` | ✅ Gravar 500 frames para dataset |
| `run_bot.bat --no-overlay` | ✅ Sem overlay debug |
| `run_bot.bat --debug` | ✅ Guardar screenshots de debug |
| `python -m soberana.autonomous` | ✅ Modo autónomo direto (mesmo que run_bot) |
| `python -m soberana.autonomous --pvpmode` | ✅ PvP direto |
| `python -m soberana.autonomous --replay <file>` | ✅ Replay de episódio |
| `python -m soberana.autonomous --export-yolo <dir>` | ✅ Exportar dataset YOLO |
| `python train_soberana.py` | ✅ Treinar YOLO (dataset + fine-tune) |
| `python train_soberana.py --export data/episodes/` | ✅ Exportar + treinar |
| `run_monitor.bat` | ✅ Auto-restart watchdog (crash recovery) |

---

## 🎯 YOLO Detection

### O que o YOLO Deteta:

| Classe | Descrição | Uso |
|--------|-------------|-----|
| **enemy** | Inimigos (oponentes) | Evitar/combat |
| **ally** | Aliados (modo team) | Cooperar |
| **resource** | Recursos (caixas, power-ups) | Colecionar |
| **obstacle** | Obstáculos (paredes, rochas) | Evitar |
| **goal** | Baliza (modo Brawl Ball) | Atacar/Defender |

### Treinar YOLO (Custom Dataset)

```powershell
# 1. Gravar replays
.\run_bot.bat --record 1000

# 2. Exportar dataset YOLO
python -m soberana.autonomous --export-yolo data/episodes/

# 3. Treinar modelo
python train_soberana.py --epochs 100 --batch-size 16

# 4. Avaliar modelo
python train_soberana.py --val
```

---

## 🤖 Adaptive Decision (FSM)

### Estados do FSM:

| Estado | Descrição | Transição |
|--------|-------------|-------------|
| **IDLE** | À espera (menu) | → `FIND_MATCH` |
| **FIND_MATCH** | Procurar partida | → `WAIT_MATCH` |
| **WAIT_MATCH** | À espera de match | → `PLAYING` |
| **PLAYING** | A jogar | → `WIN` ou `LOSE` ou `DRAW` |
| **WIN** | Vitória | → `IDLE` |
| **LOSE** | Derrota | → `IDLE` |
| **DRAW** | Empate | → `IDLE` |
| **DISCONNECTED** | Desconexão | → `RECONNECT` |

---

## 🧑‍💻 Humanization (Anti-Ban)

### Técnicas:

1. **Delays Aleatórios:**
   - Tempo entre ações: 0.5s - 2.0s
   - Movimento não-linear (curva Bezier)

2. **Padrões Humanos:**
   - Erros ocasionais (miss-click)
   - Pausas "humanas" (AFK simulado)

3. **Rotação de IP (Opcional):**
   - VPN (não recomendado para contas reais)
   - Proxy (se disponível)

---

## 📊 Dataset & Training

### Estrutura do Dataset (YOLO):

```
data/datasets/
├── images/
│   ├── train/      # Imagens de treino
│   ├── val/        # Imagens de validação
│   └── test/       # Imagens de teste
├── labels/
│   ├── train/      # Labels (YOLO format)
│   ├── val/
│   └── test/
└── data.yaml        # Config YOLO
```

### Treinar YOLO:

```powershell
# Treino completo
python train_soberana.py --epochs 100 --batch-size 16 --img-size 640

# Fine-tune (modelo pré-treinado)
python train_soberana.py --weights models/best.pt --epochs 50

# Validar
python train_soberana.py --val --weights models/best.pt
```

---

## 🔧 Configuração (`config/config.yaml`)

```yaml
# ADB Settings
adb:
  host: "127.0.0.1"
  port: 5555  # BlueStacks default

# YOLO Settings
yolo:
  model: "models/best.pt"
  conf_threshold: 0.25
  iou_threshold: 0.45

# Joystick Settings
joystick:
  center_x: 500
  center_y: 800
  radius: 100

# Humanization
humanize:
  min_delay: 0.5
  max_delay: 2.0
  bezier_curves: true
  miss_click_rate: 0.05  # 5% chance

# Game Settings
game:
  mode: "pvpmode"  # pvpmode / coop / solo
  max_matches: 10
  max_duration: 3600  # 1 hora
```

---

## 🐛 Troubleshooting

### ADB não coneta

**Causa:** BlueStacks ADB não ativado.

**Solução:**
1. BlueStacks → Configurações → Preferências → **ADB** (ativar)
2. Reiniciar BlueStacks
3. Verificar: `adb devices`

### YOLO não deteta nada

**Causa:** Modelo não treinado ou dataset fraco.

**Solução:**
1. Gravar mais replays (`--record 1000`)
2. Exportar dataset (`--export-yolo`)
3. Treinar modelo (`train_soberana.py`)

### Joystick não funciona

**Causa:** Calibração incorreta.

**Solução:**
```powershell
python -m soberana.autonomous --calibrate
```

### Watchdog não reinicia

**Causa:** Permissões ou caminho incorreto.

**Solução:**
```powershell
# Verificar logs
.\run_monitor.bat > logs\monitor.log 2>&1
```

---

## 🧪 Testes

```powershell
# Testes unitários
.\venv\Scripts\python.exe -m pytest tests/ -v

# Testar deteção YOLO
python -m soberana.vision --test

# Testar decisões FSM
python -m soberana.decision --test

# Testar controlo de joystick
python -m soberana.control --test
```

---

## 📂 Estrutura do Projeto

```
bot brawl/
├── soberana/              # Código principal
│   ├── autonomous.py    # Entry point
│   ├── vision.py        # YOLO detection
│   ├── decision.py      # FSM
│   ├── control.py       # Joystick
│   ├── humanize.py     # Anti-ban
│   ├── replay.py       # Recording
│   ├── export_yolo.py  # Dataset export
│   └── utils.py        # Utilitários
├── scripts/               # Launchers
│   ├── run_bot.bat
│   ├── run_monitor.bat
│   └── train_soberana.py
├── config/                # Configurações
│   ├── config.yaml
│   └── joystick.json
├── data/                  # Dados
│   ├── episodes/        # Replays
│   └── datasets/        # YOLO datasets
├── models/                # Modelos YOLO
│   └── best.pt
├── tests/                 # Testes
├── logs/                  # Logs
├── .env.example          # Template ambiente
├── requirements.txt       # Dependências
└── README.md            # Este ficheiro
```

---

## 📝 Licença

MIT — usar com responsabilidade.

⚠️ **Aviso Legal:** Bots em jogos online violam os Termos de Serviço. Usar **apenas em contas de teste**. Contas reais podem ser **banidas permanentemente**.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Por favor:

1. Fazer fork do repositório
2. Criar uma branch de funcionalidade
3. Fazer as tuas alterações
4. Submeter um Pull Request

---

## 📞 Suporte

Para problemas e questões:

- Consultar a secção de troubleshooting
- Rever logs em `logs/soberana.log`
- Abrir uma issue no GitHub

---

## 🙏 Agradecimentos

- **Ultralytics YOLO:** https://github.com/ultralytics/ultralytics
- **OpenCV:** https://opencv.org/
- **BlueStacks:** https://www.bluestacks.com/
- **ADB (Android Debug Bridge):** https://developer.android.com/studio/command-line/adb

---

## 📈 Estatísticas do Projeto

- **Última atualização:** 2026-06-28
- **Branch:** `master`
- **Total de ficheiros:** ~50 (código fonte)
- **Módulos Python:** 10+
- **Cobertura de testes:** 70%+
- **YOLO mAP:** 0.65 (dataset próprio)

---

**Feito com ❤️ em Portugal** 🇵🇹🇵🇬
