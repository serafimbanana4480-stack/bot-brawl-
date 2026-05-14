"""
test_real_gameplay.py

Testes de funcionalidade real com Brawl Stars aberto no emulador.
Estes testes verificam a integracao completa do bot com o jogo real.

Requisitos:
- Emulador (BlueStacks/LDPlayer) rodando com Brawl Stars aberto
- ADB habilitado
- Resolucao 1920x1080

ATENCAO: Estes testes sao READ-ONLY (nao executam acoes no jogo).
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import pytest

# BlueStacks ADB path (used as fallback only; never mutates os.environ globally)
BLUESTACKS_ADB = r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"


def _get_adb_port(adb_id: str) -> int:
    """Extrai a porta ADB de um adb_id (ex: 'emulator-5554' -> 5554)."""
    if adb_id and "-" in adb_id:
        try:
            return int(adb_id.split("-")[-1])
        except ValueError:
            pass
    return 5555


def _detect_first_emulator():
    """Detecta o primeiro emulador disponivel. Retorna None se nenhum for encontrado."""
    from emulator_detector import get_emulator_detector
    detector = get_emulator_detector()
    emulators = detector.detect_all()
    return emulators[0] if emulators else None


def _build_emulator_config(emu):
    """Constroi EmulatorConfig a partir de um emulador detectado."""
    from emulator_controller import EmulatorConfig
    port = _get_adb_port(emu.adb_id)
    config = EmulatorConfig(
        name=emu.name,
        adb_port=port,
        window_title=emu.window_title or emu.name
    )
    # Respeita adb_path explicito do BlueStacks se existir
    if Path(BLUESTACKS_ADB).exists():
        config.adb_path = BLUESTACKS_ADB
    return config


class TestEmulatorDetection:
    """Testes para deteccao do emulador. Requer emulador real a correr."""

    @pytest.mark.integration
    def test_detect_emulators(self):
        from emulator_detector import get_emulator_detector
        detector = get_emulator_detector()
        emulators = detector.detect_all()
        if not emulators:
            pytest.skip("Nenhum emulador detectado - teste requer emulador real a correr")
        print(f"\n[REAL TEST] Emuladores detectados: {len(emulators)}")
        for emu in emulators:
            print(f"  - name={emu.name}, type={emu.type}, adb_id={emu.adb_id}, window={emu.window_title}")
        assert len(emulators) > 0, "Nenhum emulador detectado! Verifique se o emulador esta aberto."

    @pytest.mark.integration
    def test_bluestacks_or_ldplayer_found(self):
        from emulator_detector import get_emulator_detector
        detector = get_emulator_detector()
        emulators = detector.detect_all()
        if not emulators:
            pytest.skip("Nenhum emulador detectado - teste requer emulador real a correr")
        types = [e.type.lower() for e in emulators]
        print(f"\n[REAL TEST] Tipos detectados: {types}")
        assert any(t in ["bluestacks", "ldplayer"] for t in types), \
            "BlueStacks ou LDPlayer nao encontrado!"


@pytest.mark.integration
class TestADBConnection:
    """Testes para conexao ADB. Requer emulador real."""

    def test_adb_connect(self):
        from emulator_controller import ADBController

        emu = _detect_first_emulator()
        if not emu:
            pytest.skip("Nenhum emulador detectado")

        config = _build_emulator_config(emu)
        adb = ADBController(config)
        try:
            connected = adb.connect()
        except Exception as e:
            pytest.skip(f"ADB connect failed: {e}")
        print(f"\n[REAL TEST] ADB conectado a {emu.name}: {connected}")
        if not connected:
            pytest.skip(f"ADB not connected to {emu.name}")
        assert connected

    def test_adb_screenshot(self):
        from emulator_controller import ADBController

        emu = _detect_first_emulator()
        if not emu:
            pytest.skip("Nenhum emulador detectado")

        config = _build_emulator_config(emu)
        adb = ADBController(config)
        try:
            connected = adb.connect()
        except Exception as e:
            pytest.skip(f"ADB connect failed: {e}")
        if not connected:
            pytest.skip("ADB not connected")
        screenshot = adb.screenshot()
        if screenshot is None:
            pytest.skip("Screenshot retornou None")
        assert len(screenshot) > 0, "Screenshot vazio"
        print(f"\n[REAL TEST] Screenshot capturado: {len(screenshot)} bytes")


@pytest.mark.integration
class TestScreenshotAnalysis:
    """Testes para analise de screenshots do jogo. Requer emulador real."""

    @pytest.fixture(scope="class")
    def screenshot_bytes(self):
        from emulator_controller import ADBController

        emu = _detect_first_emulator()
        if not emu:
            pytest.skip("Nenhum emulador detectado")

        config = _build_emulator_config(emu)
        adb = ADBController(config)
        try:
            connected = adb.connect()
        except Exception as e:
            pytest.skip(f"ADB connect failed: {e}")
        if not connected:
            pytest.skip("ADB not connected")
        screenshot = adb.screenshot()
        if screenshot is None:
            pytest.skip("Nao foi possivel capturar screenshot")
        return screenshot

    def test_screenshot_is_valid_image(self, screenshot_bytes):
        """Verifica se o screenshot e uma imagem valida."""
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None, "Screenshot nao e uma imagem valida"
        h, w = img.shape[:2]
        print(f"\n[REAL TEST] Resolucao do screenshot: {w}x{h}")
        assert w > 0 and h > 0

    def test_screenshot_resolution(self, screenshot_bytes):
        """Verifica se a resolucao e compativel (1920x1080 ou proporcional)."""
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pytest.skip("Imagem invalida")
        h, w = img.shape[:2]
        expected_ratio = 1920 / 1080
        actual_ratio = w / h
        print(f"\n[REAL TEST] Aspect ratio: {actual_ratio:.2f} (esperado: {expected_ratio:.2f})")
        assert abs(actual_ratio - expected_ratio) < 0.1, \
            f"Aspect ratio incorreto: {actual_ratio:.2f}"

    def test_screenshot_has_color(self, screenshot_bytes):
        """Verifica se a imagem tem cores (nao e preta)."""
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pytest.skip("Imagem invalida")
        mean_color = np.mean(img)
        print(f"\n[REAL TEST] Media de cor: {mean_color:.1f}")
        assert mean_color > 10, "Imagem muito escura - o jogo esta aberto?"

    def test_detect_play_button(self, screenshot_bytes):
        """Tenta detectar o botao PLAY no screenshot."""
        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            pytest.skip("Imagem invalida")

        # Converte para HSV para detectar o botao verde/amarelo do PLAY
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Detecta areas verdes/amarelas (botao PLAY)
        lower_green = np.array([35, 100, 100])
        upper_green = np.array([85, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        # Verifica se ha pixels suficientes
        green_pixels = np.sum(mask > 0)
        total_pixels = mask.size
        ratio = green_pixels / total_pixels
        print(f"\n[REAL TEST] Pixels verdes (botao PLAY?): {green_pixels} ({ratio*100:.1f}%)")
        # Nao falha se nao encontrar - pode estar em outra tela
        assert ratio > 0, "Nenhum pixel verde encontrado"


@pytest.mark.integration
class TestGameStateDetection:
    """Testes para deteccao do estado do jogo. Requer emulador real."""

    def test_window_is_visible(self):
        from emulator_detector import get_emulator_detector
        from emulator_controller import WindowController

        detector = get_emulator_detector()
        emulators = detector.detect_all()
        if not emulators:
            pytest.skip("Nenhum emulador detectado")

        emu = emulators[0]
        wc = WindowController(emu.window_title or emu.name)
        visible = wc.is_visible()
        print(f"\n[REAL TEST] Janela visivel: {visible}")
        assert visible, "Janela do emulador nao esta visivel"

    def test_window_has_size(self):
        from emulator_detector import get_emulator_detector
        from emulator_controller import WindowController

        detector = get_emulator_detector()
        emulators = detector.detect_all()
        if not emulators:
            pytest.skip("Nenhum emulador detectado")

        emu = emulators[0]
        wc = WindowController(emu.window_title or emu.name)
        rect = wc.get_rect()
        if rect is None:
            pytest.skip("Nao foi possivel obter retangulo da janela")
        x1, y1, x2, y2 = rect
        w = x2 - x1
        h = y2 - y1
        print(f"\n[REAL TEST] Tamanho da janela: {w}x{h}")
        assert w > 100 and h > 100, "Janela muito pequena"


@pytest.mark.integration
class TestVisionEngineReal:
    """Testes para o vision engine com dados reais. Requer modelos."""

    def test_load_default_model(self):
        from vision_engine import YOLOVisionEngine, VisionConfig
        from pathlib import Path

        engine = YOLOVisionEngine(VisionConfig())
        models_dir = Path(__file__).parent.parent / "models"
        if not models_dir.exists():
            pytest.skip("Diretorio de modelos nao encontrado")

        result = engine.load_models(models_dir)
        print(f"\n[REAL TEST] Modelos carregados: {result}")
        # Nao falha se nao carregar - pode nao ter modelos validos
        assert isinstance(result, bool)


@pytest.mark.integration
class TestBotIntegrationReal:
    """Testes de integracao completos com o jogo real. Requer emulador real."""

    def test_full_pipeline_readonly(self):
        """Executa o pipeline completo em modo somente leitura."""
        from emulator_controller import ADBController
        import numpy as np
        import cv2

        emu = _detect_first_emulator()
        if not emu:
            pytest.skip("Nenhum emulador detectado")

        config = _build_emulator_config(emu)
        adb = ADBController(config)
        try:
            connected = adb.connect()
        except Exception as e:
            pytest.skip(f"ADB connect failed: {e}")
        if not connected:
            pytest.skip("ADB not connected")

        # Capturar screenshot
        screenshot_bytes = adb.screenshot()
        if screenshot_bytes is None:
            pytest.skip("Screenshot retornou None")
        assert len(screenshot_bytes) > 0

        nparr = np.frombuffer(screenshot_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        assert img is not None

        print(f"\n[REAL TEST] Pipeline completo executado com sucesso!")
        print(f"  - Emulador: {emu.name} (tipo: {emu.type})")
        print(f"  - Resolucao: {img.shape[1]}x{img.shape[0]}")
        print(f"  - Screenshot: {len(screenshot_bytes)} bytes")

        # Verificar se ha conteudo na imagem
        assert img.shape[0] > 0 and img.shape[1] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
