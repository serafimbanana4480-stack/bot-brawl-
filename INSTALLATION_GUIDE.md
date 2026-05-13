# Guia de Instalação e Troubleshooting - Soberana Omega Brawl Stars Bot

## 📋 Índice
1. [Requisitos do Sistema](#requisitos)
2. [Instalação Passo a Passo](#instalacao)
3. [Configuração do Emulador](#emulador)
4. [Configuração do Bot](#configuracao)
5. [Teste de Funcionamento](#teste)
6. [Troubleshooting Comum](#troubleshooting)
7. [Solução de Problemas Específicos](#problemas-especificos)
8. [FAQ](#faq)

---

## 🔧 Requisitos do Sistema

### Hardware Mínimo
- **CPU**: Intel i5 8ª geração ou equivalente (recomendado: i7/Ryzen 5)
- **RAM**: 8GB mínimo (recomendado: 16GB)
- **GPU**: Não obrigatória, mas recomendada para YOLO (NVIDIA GTX 1060 ou superior)
- **Armazenamento**: 10GB livres

### Software
- **Sistema Operacional**: Windows 10/11 (recomendado), Linux (compatível)
- **Python**: 3.9 - 3.11
- **Emulador Android**: LDPlayer 4/5 ou BlueStacks 5

### Dependências Principais
- OpenCV >= 4.8.0
- PyTorch >= 2.0.0
- Ultralytics (YOLOv8) >= 8.0.0
- EasyOCR (opcional, para detecção OCR)
- ADB (Android Debug Bridge)

---

## 📥 Instalação Passo a Passo

### 1. Clonar o Repositório

```bash
git clone https://github.com/seu-repo/soberana-omega.git
cd soberana-omega
```

### 2. Criar Ambiente Virtual

```bash
# Usando venv
python -m venv venv

# Ativar no Windows
venv\Scripts\activate

# Ativar no Linux/Mac
source venv/bin/activate
```

### 3. Instalar Dependências

```bash
# Instalar dependências básicas
pip install -r requirements.txt

# Instalar EasyOCR (opcional, para detecção OCR)
pip install easyocr

# Instalar PyTorch com suporte CUDA (se tiver GPU NVIDIA)
# Visite https://pytorch.org/ para instruções específicas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 4. Configurar ADB

#### Windows
```bash
# Download ADB Platform Tools
# https://developer.android.com/studio/releases/platform-tools

# Extrair para C:\platform-tools
# Adicionar ao PATH do sistema

# Verificar instalação
adb version
```

#### Linux
```bash
sudo apt-get install android-tools-adb
```

### 5. Configurar Emulador

#### LDPlayer
1. Baixar e instalar LDPlayer 4 ou 5
2. Abrir configurações do LDPlayer
3. Ir em "Settings" > "Other settings"
4. Habilitar "Enable ADB"
5. Anotar a porta ADB (default: 5555)

#### BlueStacks
1. Baixar e instalar BlueStacks 5
2. Abrir configurações do BlueStacks
3. Ir em "Advanced" > "Android Debug Bridge"
4. Habilitar "Enable Android Debug Bridge"
5. Anotar a porta ADB (default: 5555)

### 6. Testar Conexão ADB

```bash
# Listar dispositivos conectados
adb devices

# Deve mostrar algo como:
# List of devices attached
# emulator-5555   device

# Se não aparecer, verificar configuração do emulador
```

---

## 🎮 Configuração do Emulador

### Configurações Recomendadas

#### Resolução
- **Resolução**: 1920x1080
- **DPI**: 280 (ou 320)
- **FPS**: 60 (ou 30 se o PC for fraco)

#### Configurações de Performance
- **CPU**: 4 cores (ou mais se disponível)
- **RAM**: 4096MB (ou mais se disponível)
- **GPU**: Habilitar aceleração de hardware

#### Configurações do Brawl Stars
- **Gráficos**: Médio ou Alto (não Ultra)
- **FPS**: 60
- **Modo de jogo**: Gem Grab (para testes iniciais)

---

## ⚙️ Configuração do Bot

### 1. Configurar config.json

```json
{
    "version": "1.0.0",
    "game": {
        "mode": "gem_grab",
        "brawler": "colt",
        "language": "en",
        "resolution": "1920x1080"
    },
    "emulator": {
        "type": "ldplayer",  // ou "bluestacks"
        "adb_port": 5555,
        "window_title": "LDPlayer",  // ou "BlueStacks App Player"
        "resolution": [1920, 1080],
        "dpi": 280
    },
    "safety": {
        "max_trophies": 400,
        "max_session_hours": 3.0,
        "min_apm": 20,
        "max_apm": 60
    },
    "vision": {
        "main_model": "brawlstars_yolov8.pt",
        "confidence_threshold": 0.37
    },
    "dashboard": {
        "enabled": true,
        "port": 8765
    }
}
```

### 2. Baixar Modelos YOLO

```bash
# Criar diretório de modelos
mkdir models

# Baixar modelo pré-treinado
# (ou treinar seu próprio modelo - veja documentação de treinamento)
wget https://seu-modelo-url/brawlstars_yolov8.pt -O models/brawlstars_yolov8.pt
```

### 3. Criar Templates para Calibração

```bash
# Criar diretório de templates
mkdir -p images/templates

# Capturar screenshots dos elementos principais:
# - Botão PLAY
# - Logo Brawl Stars
# - Ícone de troféu
# - Botão X (fechar)
# - Botão PROCEED
# - Botão PLAY AGAIN

# Salvar como PNG em images/templates/
```

### 4. Executar Calibração Inicial

```python
from pylaai_real.auto_calibrator import interactive_calibration_setup
from pylaai_real.screenshot_taker import ScreenshotTaker

# Criar screenshot taker
screenshot = ScreenshotTaker()

# Elementos para calibrar
elements = [
    "play_button",
    "brawl_stars_logo",
    "trophy_icon",
    "x_button"
]

# Executar calibração interativa
coords = interactive_calibration_setup(screenshot, elements)

# Salvar coordenadas
import json
with open("data/calibrated_coords.json", "w") as f:
    json.dump(coords, f, indent=2)
```

---

## 🧪 Teste de Funcionamento

### 1. Teste de Captura de Tela

```python
from pylaai_real.screenshot_taker import ScreenshotTaker
import cv2

# Criar screenshot taker
screenshot = ScreenshotTaker()

# Capturar tela
img = screenshot.capture()

# Mostrar imagem
cv2.imshow("Screenshot", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Captura de tela funcionando!")
```

### 2. Teste de Detecção YOLO

```python
from ultralytics import YOLO
import cv2

# Carregar modelo
model = YOLO("models/brawlstars_yolov8.pt")

# Carregar imagem
img = cv2.imread("test_screenshot.png")

# Executar detecção
results = model(img)

# Mostrar resultados
for result in results:
    result.show()

print("Detecção YOLO funcionando!")
```

### 3. Teste de Detecção de Estado

```python
from pylaai_real.unified_state_detector import UnifiedStateDetector
from pylaai_real.screenshot_taker import ScreenshotTaker

# Criar componentes
screenshot = ScreenshotTaker()
detector = UnifiedStateDetector()

# Detectar estado
img = screenshot.capture()
state = detector.detect(img)

print(f"Estado detectado: {state.state} (confiança: {state.confidence:.2f})")
```

### 4. Teste Completo do Bot

```python
from brawl_bot.wrapper import PylaAIEnhanced

# Criar bot
bot = PylaAIEnhanced()

# Iniciar bot
bot.start()

# Deixar rodar por alguns minutos
import time
time.sleep(60)

# Parar bot
bot.stop()

print("Bot testado com sucesso!")
```

---

## 🔧 Troubleshooting Comum

### Problema: ADB não detecta emulador

**Sintomas:**
```
adb devices
List of devices attached
(empty)
```

**Soluções:**

1. **Verificar se ADB está habilitado no emulador**
   - LDPlayer: Settings > Other settings > Enable ADB
   - BlueStacks: Settings > Advanced > Enable Android Debug Bridge

2. **Verificar porta ADB**
   ```bash
   # Tentar conectar manualmente
   adb connect localhost:5555
   ```

3. **Reiniciar ADB**
   ```bash
   adb kill-server
   adb start-server
   ```

4. **Verificar firewall**
   - Adicionar exceção para ADB no firewall do Windows

### Problema: Captura de tela falha

**Sintomas:**
```
Error: Failed to capture screenshot
```

**Soluções:**

1. **Verificar se emulador está em foco**
   - O emulador deve estar visível e em foco

2. **Tentar método alternativo**
   - Editar config.json para usar `screenshot_method: "adb"`

3. **Verificar permissões**
   - Executar como administrador

### Problema: YOLO não detecta nada

**Sintomas:**
```
No detections found
Confidence too low
```

**Soluções:**

1. **Verificar modelo**
   ```bash
   # Verificar se modelo existe
   ls -lh models/brawlstars_yolov8.pt
   ```

2. **Ajustar threshold**
   ```json
   {
       "vision": {
           "confidence_threshold": 0.3  // Reduzir para 0.3
       }
   }
   ```

3. **Verificar iluminação**
   - O jogo deve estar visível e com boa iluminação

4. **Retreinar modelo**
   - Coletar novos dados do jogo atual
   - Retreinar YOLO com dados recentes

### Problema: Bot fica preso em "unknown"

**Sintomas:**
```
State: unknown
Confidence: 0.0
```

**Soluções:**

1. **Habilitar modo debug**
   ```python
   from pylaai_real.debug_visualizer import DebugVisualizer, DebugMode
   
   visualizer = DebugVisualizer(mode=DebugMode.DETAILED)
   visualizer.start()
   ```

2. **Verificar coordenadas**
   - As coordenadas podem ter mudado com atualização do jogo
   - Executar calibração novamente

3. **Habilitar OCR**
   ```python
   from pylaai_real.ocr_state_detector import OCRStateDetector
   
   ocr_detector = OCRStateDetector()
   state, conf = ocr_detector.detect_state_from_text(screenshot)
   ```

4. **Verificar sistema de recuperação**
   - O sistema de recuperação deve tentar voltar ao lobby automaticamente

### Problema: Bot não clica nos botões

**Sintomas:**
```
Bot detecta estado mas não executa ações
```

**Soluções:**

1. **Verificar coordenadas de clique**
   - As coordenadas podem estar incorretas
   - Executar calibração interativa

2. **Testar clique manual**
   ```bash
   adb shell input tap 960 540  # Teste clique no centro
   ```

3. **Verificar humanização**
   - O sistema de humanização pode estar adicionando muito delay
   - Reduzir delays em config.json

### Problema: Performance baixa (FPS baixo)

**Sintomas:**
```
FPS: 5-10
Cycle time: >200ms
```

**Soluções:**

1. **Reduzir resolução do emulador**
   - Mudar para 1280x720

2. **Desabilitar recursos pesados**
   ```json
   {
       "vision": {
           "enable_ocr": false
       },
       "combat": {
           "enable_prediction": false
       }
   }
   ```

3. **Usar GPU**
   - Instalar PyTorch com CUDA
   - Configurar YOLO para usar GPU

### Problema: Bot é banido

**Sintomas:**
```
Account banned
Suspicious activity detected
```

**Soluções:**

1. **Aumentar humanização**
   ```json
   {
       "humanization": {
           "random_delays": true,
           "min_delay_ms": 100,
           "max_delay_ms": 300
       }
   }
   ```

2. **Reduzir APM**
   ```json
   {
       "safety": {
           "max_apm": 40  // Reduzir de 60 para 40
       }
   }
   ```

3. **Limitar sessão**
   ```json
   {
       "safety": {
           "max_session_hours": 2.0  // Reduzir de 3.0
       }
   }
   ```

4. **Usar múltiplas contas**
   - Alternar entre contas para diluir o risco

---

## 🎯 Solução de Problemas Específicos

### Atualização do Brawl Stars

Quando o jogo é atualizado, a interface pode mudar.

**Passos para corrigir:**

1. **Invalidar cache de coordenadas**
   ```python
   from pylaai_real.auto_calibrator import AutoCalibrator
   
   calibrator = AutoCalibrator()
   calibrator.invalidate_cache()  # Invalidar todo o cache
   ```

2. **Recapturar templates**
   - Capturar novos screenshots dos elementos
   - Substituir templates antigos

3. **Retreinar YOLO (se necessário)**
   - Coletar novos dados
   - Retreinar modelo

4. **Executar calibração novamente**
   ```python
   coords = interactive_calibration_setup(screenshot, elements)
   ```

### Mudança de Emulador

Se mudar de LDPlayer para BlueStacks (ou vice-versa):

1. **Atualizar config.json**
   ```json
   {
       "emulator": {
           "type": "bluestacks",  // Mudar para novo emulador
           "window_title": "BlueStacks App Player"
       }
   }
   ```

2. **Verificar porta ADB**
   - Pode ser diferente (5555 vs 5556)

3. **Recalibrar coordenadas**
   - A resolução pode ser diferente

### Erro de Memória

**Sintomas:**
```
MemoryError
CUDA out of memory
```

**Soluções:**

1. **Reduzir batch size do YOLO**
   ```python
   model = YOLO("model.pt")
   results = model(img, batch=1)  # Reduzir batch
   ```

2. **Limpar cache periodicamente**
   ```python
   import gc
   gc.collect()
   ```

3. **Aumentar RAM virtual**
   - Aumentar pagefile do Windows

---

## ❓ FAQ

### P: O bot funciona em todos os modos de jogo?
R: Otimizado para Gem Grab, mas pode funcionar em outros modos com ajustes.

### P: Preciso de GPU?
R: Não obrigatório, mas recomendado para melhor performance. CPU funciona mas é mais lento.

### P: O bot pode me banir?
R: Existe risco, mas o sistema de humanização reduz significativamente. Use por sua conta e risco.

### P: Posso usar múltiplas contas?
R: Sim, mas recomenda-se alternar para reduzir risco.

### P: Com que frequência devo atualizar o bot?
R: Após cada atualização importante do Brawl Stars.

### P: O bot funciona no celular real?
R: Não, apenas em emuladores PC via ADB.

### P: Posso contribuir com o projeto?
R: Sim! Pull requests são bem-vindos.

### P: Onde posso conseguir ajuda?
R: Verifique o GitHub Issues, Discord do projeto, ou documentação.

---

## 📞 Suporte

Se você encontrar problemas não cobertos neste guia:

1. **Verifique logs**: `logs/brawl_bot.log`
2. **Habilite modo debug**: Use `DebugVisualizer` para ver o que o bot está detectando
3. **Colete informações**: Sistema, emulador, versão do Python, etc.
4. **Abra uma issue**: No GitHub com detalhes do problema

---

## 📝 Notas Importantes

- **Use por sua conta e risco**: Botting pode violar termos de serviço do jogo
- **Não abuse**: Limite sessões e use humanização adequada
- **Mantenha atualizado**: Atualize o bot regularmente
- **Faça backup**: Backup de config.json e modelos antes de atualizações
- **Teste primeiro**: Teste em conta secundária antes de usar conta principal

---

**Última atualização**: 2024
**Versão do documento**: 1.0
