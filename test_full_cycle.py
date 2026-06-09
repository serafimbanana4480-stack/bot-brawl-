"""
test_full_cycle.py

Teste de integração do ciclo completo do bot.
Simula o fluxo: lobby -> play -> in_game sem iniciar threads.
"""

import sys

import time
import numpy as np
from PIL import Image
from pathlib import Path

print("=" * 60)
print("TESTE CICLO COMPLETO DO BOT")
print("=" * 60)

# 1. Screenshot
print("\n[1/8] Capturando screenshot...")
try:
    from pylaai_real.screenshot_taker import ScreenshotTaker
    taker = ScreenshotTaker("LDPlayer")
    if taker.find_window():
        img = taker.take()
        print(f"  OK: {img.shape[1]}x{img.shape[0]}")
        screenshot = img
    else:
        print("  FALHA: Janela não encontrada. Usando screenshot salvo...")
        screenshot = np.array(Image.open('c:/Users/rodri/Desktop/bot brawl/screenshot_current.png'))
except Exception as e:
    print(f"  ERRO: {e}")
    screenshot = None

if screenshot is None:
    print("\n[ABORTAR] Sem screenshot disponível")
    sys.exit(1)

# 2. Detector de estado
print("\n[2/8] Testando detector de estado...")
try:
    from pylaai_real.unified_state_detector import UnifiedStateDetector
    det = UnifiedStateDetector(images_path=Path("images"))
    result = det.detect(screenshot)
    print(f"  Estado: {result.state} (conf={result.confidence:.2f})")
    print(f"  Botão: {result.button_coords}")
    if result.state == "lobby":
        print("  OK: Lobby detetado!")
    else:
        print(f"  ALERTA: Estado={result.state}")
except Exception as e:
    print(f"  ERRO: {e}")
    sys.exit(1)

# 3. Screenshot Analyzer
print("\n[3/8] Testando screenshot analyzer...")
try:
    from core.screenshot_analyzer import ScreenshotAnalyzer
    analyzer = ScreenshotAnalyzer()
    analysis = analyzer.analyze(screenshot)
    print(f"  Válido: {analysis.valid}")
    print(f"  Color space: {analysis.color_space}")
    print(f"  Play button amarelo: {analysis.region_health.get('play_button_yellow', 0):.2f}")
except Exception as e:
    print(f"  ERRO: {e}")

# 4. Lobby Automator (sem ADB)
print("\n[4/8] Testando lobby automator...")
try:
    from pylaai_real.lobby_automator import LobbyAutomator, BrawlerQueue, BrawlerConfig

    class MockCtrl:
        def __init__(self):
            self.actions = []
        def tap_scaled(self, x, y):
            self.actions.append(('tap', x, y))
        def keyevent(self, code):
            self.actions.append(('key', code))

    queue = BrawlerQueue()
    queue.add_brawler(BrawlerConfig(name="shelly", target_trophies=100))
    ctrl = MockCtrl()
    lobby = LobbyAutomator(queue, ctrl, window_w=1920, window_h=1080,
                           images_path=str(Path("images")))
    lobby.set_state_detector(det)
    lobby.set_screenshot_func(lambda: screenshot)

    # Testar press_play
    print("  A executar press_play...")
    result = lobby.press_play()
    print(f"  Resultado: {result}")
    print(f"  Ações: {ctrl.actions}")
    if any(a[0] == 'tap' for a in ctrl.actions):
        print("  OK: Bot clicou no Play!")
    else:
        print("  FALHA: Bot não clicou em nada")
except Exception as e:
    print(f"  ERRO: {e}")
    import traceback
    traceback.print_exc()

# 5. Modelo YOLO
print("\n[5/8] Testando modelo YOLO...")
try:
    from ultralytics import YOLO
    model = YOLO('c:/Users/rodri/Desktop/bot brawl/models/brawlstars_yolov8.pt')
    results = model(screenshot, conf=0.1, verbose=False)
    for r in results:
        print(f"  Detecções: {len(r.boxes)}")
        for b in r.boxes[:3]:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            name = r.names.get(cls, f'class_{cls}')
            print(f"    {name}: {conf:.2f}")
except Exception as e:
    print(f"  ERRO: {e}")

# 6. AutoFixEngine
print("\n[6/8] Testando AutoFixEngine...")
try:
    from core.auto_fix_engine import AutoFixEngine
    engine = AutoFixEngine(
        screenshot_func=lambda: screenshot,
        click_func=lambda x, y: None,
        key_func=lambda k: None,
        state_detector=det,
    )
    forced = engine.tick("lobby")
    print(f"  Estado forçado: {forced}")
    print(f"  Status: {engine.get_status()}")
except Exception as e:
    print(f"  ERRO: {e}")

# 7. StateManager (sem threads)
print("\n[7/8] Testando StateManager ciclo único...")
try:
    from pylaai_real.state_manager import StateManager
    from pylaai_real.state_finder import StateFinder

    sf = StateFinder(str(Path("images")))
    sm = StateManager(
        screenshot_taker=taker,
        state_finder=sf,
        lobby_automator=lobby,
        play_logic=None,
        emulator_controller=ctrl,
        movement=None,
        unified_state_detector=det,
        auto_fix_engine=engine,
    )
    # Simular um ciclo
    sm.current_state = 'lobby'
    sm.state_start_time = time.time()
    print(f"  Estado inicial: {sm.current_state}")
    print("  OK: StateManager inicializado")
except Exception as e:
    print(f"  ERRO: {e}")

# 8. Resumo
print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)
print("O bot está configurado para:")
print("- Detetar o lobby (UnifiedStateDetector corrigido)")
print("- Clicar no Play (LobbyAutomator + detector)")
print("- Entrar em jogo e detetar inimigos (YOLO)")
print("- Auto-recovery se ficar preso (AutoFixEngine)")
print("\nPróximos passos:")
print("1. Execute: python wrapper.py (ou start_all.bat)")
print("2. Monitore os logs em tempo_real")
print("3. Se o bot ficar parado, execute: python diagnose_bot.py")
print("=" * 60)
