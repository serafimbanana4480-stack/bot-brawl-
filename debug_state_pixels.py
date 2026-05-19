"""Diagnosticar o que o UnifiedStateDetector vê na screenshot atual"""
import numpy as np
import cv2
from pylaai_real.screenshot_taker import ScreenshotTaker
from pylaai_real.unified_state_detector import UnifiedStateDetector, DynamicCoordinates

# Capturar screenshot
taker = ScreenshotTaker()
taker.find_window()
img = taker.take()
if img is None:
    print("Falha ao capturar screenshot")
    exit(1)

h, w = img.shape[:2]
print(f"Screenshot: {w}x{h}")

# Coordenadas dinâmicas
c = DynamicCoordinates(w, h)
print(f"\nCoordenadas calculadas:")
print(f"  play_button: {c.play_button}")
print(f"  load_button: {c.load_button}")
print(f"  defeated1: {c.defeated1}")
print(f"  defeated2: {c.defeated2}")
print(f"  player_hp_bar: {c.player_hp_bar}")
print(f"  joystick_center: {c.joystick_center}")

# Verificar cores em pontos-chave
points = {
    'play_button': c.play_button,
    'load_button': c.load_button,
    'defeated1': c.defeated1,
    'defeated2': c.defeated2,
    'player_hp_bar': c.player_hp_bar,
    'joystick_center': c.joystick_center,
    'proceed_button': c.proceed_button,
    'star_drop1': c.star_drop1,
    'star_drop2': c.star_drop2,
}

print(f"\nCores nos pontos de interesse:")
for name, (x, y) in points.items():
    if 0 <= x < w and 0 <= y < h:
        bgr = img[y, x]
        rgb = (bgr[2], bgr[1], bgr[0])
        print(f"  {name} ({x},{y}): RGB={rgb}")
    else:
        print(f"  {name} ({x},{y}): FORA DA IMAGEM!")

# Verificar região do botão Play (área grande)
px, py = c.play_button
if 0 <= px < w and 0 <= py < h:
    region = img[max(0,py-20):min(h,py+20), max(0,px-50):min(w,px+50)]
    if region.size > 0:
        print(f"\nRegião play_button (40x100):")
        print(f"  Mean RGB: {np.mean(region[:,:,2]):.0f}, {np.mean(region[:,:,1]):.0f}, {np.mean(region[:,:,0]):.0f}")
        print(f"  Std RGB: {np.std(region[:,:,2]):.0f}, {np.std(region[:,:,1]):.0f}, {np.std(region[:,:,0]):.0f}")

# Usar detector completo
det = UnifiedStateDetector(images_path='images', window_w=w, window_h=h)
result = det.detect(img)
print(f"\n=== RESULTADO DETECTOR ===")
print(f"Estado: {result.state}")
print(f"Confiança: {result.confidence:.2f}")
print(f"Método: {result.method}")

# Salvar screenshot para análise
output = img.copy()
for name, (x, y) in points.items():
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(output, name, (x+8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

cv2.imwrite('debug_state_overlay.png', cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
print("\nScreenshot com overlay salva: debug_state_overlay.png")
