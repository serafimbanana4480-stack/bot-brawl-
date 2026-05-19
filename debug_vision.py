"""
Script de diagnóstico visual - captura screenshot do emulador e mostra
o que o UnifiedStateDetector detecta.
"""
import logging
import time
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

def main():
    from pylaai_real.unified_state_detector import UnifiedStateDetector
    from pylaai_real.screenshot_taker import ScreenshotTaker
    
    from pathlib import Path
    # Inicializar
    detector = UnifiedStateDetector(images_path=Path("images"))
    screenshot = ScreenshotTaker()
    
    logger.info("=" * 60)
    logger.info("DIAGNÓSTICO VISUAL")
    logger.info("=" * 60)
    
    for i in range(5):
        logger.info(f"\n--- Captura {i+1}/5 ---")
        img = screenshot.take()
        if img is None:
            logger.error("Screenshot falhou!")
            continue
            
        logger.info(f"Screenshot: {img.shape}")
        
        # Detectar estado
        result = detector.detect(img)
        logger.info(f"Estado detectado: {result.state} (conf={result.confidence:.2f}, method={result.method})")
        logger.info(f"  Detalhes: {result.details}")
        
        # Detectar botão Play
        from pylaai_real.lobby_navigator import SmartPlayButtonDetector
        play_detector = SmartPlayButtonDetector(Path("images"))
        play_result = play_detector.find_play_button(img)
        logger.info(f"Botão Play: found={play_result.found}, coords={play_result.coords}, region={play_result.region}, conf={play_result.confidence:.2f}")
        
        # Salvar screenshot para análise manual
        import cv2
        cv2.imwrite(f"debug_screenshot_{i+1}.png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        logger.info(f"Screenshot salvo: debug_screenshot_{i+1}.png")
        
        time.sleep(2)
    
    logger.info("\n" + "=" * 60)
    logger.info("DIAGNÓSTICO COMPLETO")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()
