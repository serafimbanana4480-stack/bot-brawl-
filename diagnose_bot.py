"""
diagnose_bot.py

Script de diagnóstico autónomo para o Soberana Omega Bot.
Analisa o estado atual do jogo, o funcionamento do detector,
e fornece recomendações de correção.

Uso:
    python diagnose_bot.py

Requisitos:
    - Emulador aberto com Brawl Stars
    - Ambiente Python com dependências instaladas
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    from PIL import Image
except ImportError:
    print("[ERRO] numpy e/ou PIL não instalados. Execute: pip install numpy pillow")
    sys.exit(1)

print("=" * 60)
print("DIAGNÓSTICO SOBERANA OMEGA BOT")
print("=" * 60)

# 1. Verificar screenshot
print("\n[1/6] Verificando captura de screenshot...")
try:
    from pylaai_real.screenshot_taker import ScreenshotTaker
    taker = ScreenshotTaker("LDPlayer")
    if taker.find_window():
        img = taker.take()
        if img is not None:
            print(f"  OK: Screenshot capturado ({img.shape[1]}x{img.shape[0]})")
            screenshot = img
        else:
            print("  FALHA: Screenshot retornou None")
            screenshot = None
    else:
        print("  FALHA: Janela do emulador não encontrada")
        screenshot = None
except Exception as e:
    print(f"  ERRO: {e}")
    screenshot = None

# 2. Verificar detector de estado
print("\n[2/6] Verificando detector de estado...")
if screenshot is not None:
    try:
        from pylaai_real.unified_state_detector import UnifiedStateDetector
        det = UnifiedStateDetector(images_path=Path("images"))
        result = det.detect(screenshot)
        print(f"  Estado detetado: {result.state} (confiança: {result.confidence:.2f})")
        print(f"  Coordenadas do botão: {result.button_coords}")
        if result.state == "lobby" and result.confidence > 0.5:
            print("  OK: Lobby detetado corretamente!")
        elif result.state == "unknown":
            print("  ALERTA: Estado desconhecido - possível problema de cores ou UI diferente")
        else:
            print(f"  INFO: Estado atual é '{result.state}'")
    except Exception as e:
        print(f"  ERRO no detector: {e}")
else:
    print("  SKIP: Sem screenshot")

# 3. Verificar modelo YOLO
print("\n[3/6] Verificando modelo YOLO...")
try:
    from ultralytics import YOLO
    model_paths = [
        Path("models/brawlstars_yolov8.pt"),
        Path("models/brawlstars_yolov8_8class.pt"),
        Path("models/brawlstars_yolov8_gpu.pt"),
    ]
    model_loaded = False
    for p in model_paths:
        if p.exists():
            model = YOLO(str(p))
            print(f"  OK: Modelo carregado: {p.name}")
            if screenshot is not None:
                results = model(screenshot, conf=0.1, verbose=False)
                for r in results:
                    print(f"  Detecções no screenshot atual: {len(r.boxes)}")
            model_loaded = True
            break
    if not model_loaded:
        print("  FALHA: Nenhum modelo encontrado em models/")
except ImportError:
    print("  ERRO: ultralytics não instalado. Execute: pip install ultralytics")
except Exception as e:
    print(f"  ERRO: {e}")

# 4. Verificar ADB
print("\n[4/6] Verificando ADB...")
try:
    from emulator_controller import EmulatorConfig, EmulatorController
    config = EmulatorConfig.for_ldplayer()
    ctrl = EmulatorController(config)
    if ctrl.adb and ctrl.adb.connect():
        print("  OK: ADB conectado")
    else:
        print("  FALHA: ADB não conectado. Verifique se o emulador está aberto.")
except Exception as e:
    print(f"  ERRO: {e}")

# 5. Verificar screenshot analyzer
print("\n[5/6] Verificando screenshot analyzer...")
if screenshot is not None:
    try:
        from core.screenshot_analyzer import ScreenshotAnalyzer
        analyzer = ScreenshotAnalyzer()
        analysis = analyzer.analyze(screenshot)
        print(f"  Screenshot válido: {analysis.valid}")
        print(f"  Brilho médio: {analysis.avg_brightness:.1f}")
        print(f"  Color space: {analysis.color_space}")
        print(f"  Play button amarelo: {analysis.region_health.get('play_button_yellow', 0):.2f}")
        if analysis.issues:
            print(f"  Problemas: {analysis.issues}")
    except Exception as e:
        print(f"  ERRO: {e}")
else:
    print("  SKIP: Sem screenshot")

# 6. Verificar auto-fix engine
print("\n[6/6] Verificando auto-fix engine...")
try:
    from core.auto_fix_engine import AutoFixEngine
    print("  OK: AutoFixEngine importado")
except Exception as e:
    print(f"  ERRO: {e}")

# Recomendações
print("\n" + "=" * 60)
print("RECOMENDAÇÕES")
print("=" * 60)

if screenshot is None:
    print("- Verifique se o emulador está aberto e visível")
    print("- Verifique o título da janela em config.json")
else:
    try:
        from pylaai_real.unified_state_detector import UnifiedStateDetector
        det = UnifiedStateDetector(images_path=Path("images"))
        result = det.detect(screenshot)
        if result.state == "unknown":
            print("- O detector não reconhece o estado atual.")
            print("  Possíveis causas:")
            print("  * O jogo está numa tela diferente (popup, news, shop)")
            print("  * A resolução mudou - reinicie o bot")
            print("  * A UI do Brawl Stars foi atualizada - execute o auto-calibrador")
        elif result.state == "lobby":
            print("- O bot detetou o lobby corretamente.")
            print("- Se o bot não clicar no Play, verifique:")
            print("  * Se o AutoFixEngine está ativo no wrapper.py")
            print("  * Os logs do state_manager para erros")
        elif result.state == "in_game":
            print("- O bot detetou que está em jogo.")
            print("- Verifique se o modelo YOLO deteta inimigos no screenshot")
    except Exception:
        pass

print("\n" + "=" * 60)
print("Diagnóstico completo!")
print("=" * 60)
