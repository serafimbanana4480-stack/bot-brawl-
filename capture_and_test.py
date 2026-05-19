"""Capturar screenshot atual e testar YOLO"""
from pylaai_real.screenshot_taker import ScreenshotTaker
from ultralytics import YOLO
import cv2

# Capturar
taker = ScreenshotTaker()
taker.find_window()
img = taker.take()
if img is None:
    print("Falha ao capturar")
    exit(1)

print(f"Screenshot: {img.shape}")

# Salvar para verificação manual
cv2.imwrite('screenshot_now.png', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
print("Salva como screenshot_now.png")

# Testar YOLO
model = YOLO('models/brawlstars_yolov8_8class.pt')
print(f"Modelo classes: {model.names}")

for conf in [0.05, 0.1, 0.2, 0.3]:
    results = model(img, conf=conf, verbose=False)
    count = sum(len(r.boxes) for r in results)
    print(f"  conf={conf}: {count} detecções")
    if count > 0:
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf_val = float(box.conf[0])
                name = model.names[cls_id]
                print(f"    - {name} (id={cls_id}, conf={conf_val:.2f})")
