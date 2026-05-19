"""Testar modelo YOLO diretamente numa screenshot"""
from ultralytics import YOLO
import numpy as np

# Carregar modelo
model = YOLO('models/brawlstars_yolov8_8class.pt')
print(f"Modelo classes: {model.names}")

# Carregar screenshot
import cv2
img = cv2.imread('debug_screenshot_1.png')
if img is None:
    print("Screenshot não encontrada, capturando nova...")
    from pylaai_real.screenshot_taker import ScreenshotTaker
    taker = ScreenshotTaker()
    taker.find_window()
    img = taker.take()
    if img is not None:
        cv2.imwrite('debug_test.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        print(f"Screenshot capturada: {img.shape}")
    else:
        print("Falha ao capturar screenshot")
        exit(1)
else:
    print(f"Screenshot carregada: {img.shape}")

# Testar inferência com vários thresholds
for conf in [0.1, 0.2, 0.3, 0.4, 0.5]:
    results = model(img, conf=conf, verbose=False)
    count = 0
    for r in results:
        count += len(r.boxes)
    print(f"  conf={conf}: {count} detecções")
    if count > 0:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                name = model.names[cls_id]
                print(f"    - {name} (id={cls_id}, conf={conf_val:.2f})")
