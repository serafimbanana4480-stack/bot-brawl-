"""
Teste de funcionalidade real do bot Brawl Stars.

Requer:
- Emulador (BlueStacks) com ADB ativo
- Brawl Stars aberto
- ADB disponível

Verifica:
1. Conexão ADB
2. Screenshot do jogo
3. Detecção de estado do jogo
4. Modelos de ML
5. Ações de input
"""

import sys
import time
import logging
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent))


def test_adb_connection(adb_id: str, adb_path: str) -> bool:
    """Testa conexão ADB."""
    try:
        import subprocess
        result = subprocess.run(
            [adb_path, "-s", adb_id, "shell", "echo", "ping"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and "ping" in result.stdout:
            logger.info("✅ ADB conectado com sucesso")
            return True
    except Exception as e:
        logger.error(f"❌ ADB falhou: {e}")
    return False


def test_screenshot(adb_id: str, adb_path: str) -> bool:
    """Testa captura de screenshot via ADB."""
    try:
        import subprocess
        import numpy as np
        from PIL import Image
        import io

        result = subprocess.run(
            [adb_path, "-s", adb_id, "exec-out", "screencap", "-p"],
            capture_output=True, timeout=15
        )
        if result.returncode == 0:
            img = Image.open(io.BytesIO(result.stdout))
            arr = np.array(img)
            logger.info(f"✅ Screenshot capturado: {arr.shape}")
            # Salvar para verificação manual
            screenshot_path = Path(__file__).parent / "test_screenshot.png"
            img.save(screenshot_path)
            logger.info(f"   Screenshot salvo em: {screenshot_path}")
            return True
    except Exception as e:
        logger.error(f"❌ Screenshot falhou: {e}")
    return False


def test_brawl_stars_running(adb_id: str, adb_path: str) -> bool:
    """Verifica se Brawl Stars está em execução."""
    try:
        import subprocess
        result = subprocess.run(
            [adb_path, "-s", adb_id, "shell", "ps"],
            capture_output=True, text=True, timeout=10
        )
        if "com.supercell.brawlstars" in result.stdout:
            logger.info("✅ Brawl Stars está em execução")
            return True
        else:
            logger.warning("⚠️ Brawl Stars não está em execução")
    except Exception as e:
        logger.error(f"❌ Erro ao verificar processos: {e}")
    return False


def test_window_detection() -> bool:
    """Testa detecção da janela do emulador."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from emulator_detector import get_emulator_detector
        detector = get_emulator_detector()
        emulators = detector.detect_all()
        if emulators:
            logger.info(f"✅ Emulador detetado: {emulators[0].name} ({emulators[0].type})")
            return True
        else:
            logger.warning("⚠️ Nenhum emulador detetado")
    except Exception as e:
        logger.error(f"❌ Deteção de emulador falhou: {e}")
    return False


def test_models() -> bool:
    """Testa disponibilidade de modelos treinados."""
    try:
        models_dir = Path(__file__).parent / "models"
        pt_files = list(models_dir.glob("*.pt"))
        if pt_files:
            logger.info(f"✅ {len(pt_files)} modelos encontrados: {[f.name for f in pt_files]}")
            return True
        else:
            logger.warning("⚠️ Nenhum modelo .pt encontrado")
    except Exception as e:
        logger.error(f"❌ Erro ao verificar modelos: {e}")
    return False


def test_yolo_available() -> bool:
    """Testa se YOLO (ultralytics) está disponível."""
    try:
        from ultralytics import YOLO
        logger.info("✅ Ultralytics YOLO disponível")
        return True
    except ImportError:
        logger.warning("⚠️ Ultralytics YOLO não instalado")
    return False


def test_adb_tap(adb_id: str, adb_path: str) -> bool:
    """Testa ação de tap via ADB."""
    try:
        import subprocess
        # Tap no centro da tela (960, 540 para 1920x1080)
        result = subprocess.run(
            [adb_path, "-s", adb_id, "shell", "input", "tap", "960", "540"],
            capture_output=True, timeout=5
        )
        if result.returncode == 0:
            logger.info("✅ ADB tap funcionou")
            return True
    except Exception as e:
        logger.error(f"❌ ADB tap falhou: {e}")
    return False


def test_state_finder() -> bool:
    """Testa carregamento do StateFinder."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from pylaai_real.state_finder import StateFinder
        images_path = Path(__file__).parent / "images"
        sf = StateFinder(images_path)
        logger.info(f"✅ StateFinder inicializado com regiões: {list(sf.region_data.keys())[:5]}...")
        return True
    except Exception as e:
        logger.error(f"❌ StateFinder falhou: {e}")
    return False


def test_wrapper_initialization() -> bool:
    """Testa inicialização do wrapper principal."""
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from wrapper import PylaAIEnhanced
        bot = PylaAIEnhanced()
        logger.info("✅ PylaAIEnhanced inicializado (sem setup)")
        # Verificar componentes
        logger.info(f"   Safety: {bot.safety is not None}")
        logger.info(f"   Humanization: {bot.humanization is not None}")
        logger.info(f"   Brawler queue: {len(bot.brawler_queue.get_queue())} brawlers")
        return True
    except Exception as e:
        logger.error(f"❌ Wrapper falhou: {e}")
    return False


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Teste de funcionalidade real do bot")
    parser.add_argument("--adb-id", default="emulator-5554", help="ADB device ID")
    parser.add_argument("--adb-path", default=r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe", help="Caminho para o ADB")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("TESTE DE FUNCIONALIDADE REAL - BRAWL STARS BOT")
    logger.info("=" * 60)

    results = {}

    # 1. ADB
    results["adb_connection"] = test_adb_connection(args.adb_id, args.adb_path)

    # 2. Brawl Stars running
    results["brawl_running"] = test_brawl_stars_running(args.adb_id, args.adb_path)

    # 3. Screenshot
    results["screenshot"] = test_screenshot(args.adb_id, args.adb_path)

    # 4. Window detection
    results["window"] = test_window_detection()

    # 5. Models
    results["models"] = test_models()

    # 6. YOLO
    results["yolo"] = test_yolo_available()

    # 7. ADB Tap
    results["adb_tap"] = test_adb_tap(args.adb_id, args.adb_path)

    # 8. State Finder
    results["state_finder"] = test_state_finder()

    # 9. Wrapper
    results["wrapper"] = test_wrapper_initialization()

    # Resumo
    logger.info("=" * 60)
    logger.info("RESUMO")
    logger.info("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok in results.items():
        emoji = "✅" if ok else "❌"
        logger.info(f"{emoji} {name}: {'OK' if ok else 'FALHOU'}")
    logger.info(f"\nTotal: {passed}/{total} ({passed/total*100:.0f}%)")

    if passed == total:
        logger.info("🎉 TODOS OS TESTES PASSARAM - Sistema operacional")
        return 0
    else:
        logger.info("⚠️ ALGUNS TESTES FALHARAM - Verificar logs acima")
        return 1


if __name__ == "__main__":
    sys.exit(main())
