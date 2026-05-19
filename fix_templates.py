import pathlib
p = pathlib.Path('pylaai_real/unified_state_detector.py')
text = p.read_text(encoding='utf-8')

# Baixar threshold do joystick de 0.8 para 0.55 (ainda preciso mas mais permissivo)
old = "        found, conf, pos = self._template_match(\n            image, 'joystick.png', joystick_region, threshold=0.8\n        )\n        if found and conf > 0.4:"
new = "        found, conf, pos = self._template_match(\n            image, 'joystick.png', joystick_region, threshold=0.55\n        )\n        if found and conf > 0.4:"

if old in text:
    text = text.replace(old, new)
    print('OK: joystick threshold atualizado para 0.55')
else:
    print('ERRO: joystick threshold nao encontrado')

# Baixar threshold do play button de 0.5 para 0.35
old2 = "        found, conf, pos = self._template_match(\n            image, 'play_button.png', play_region, threshold=0.5\n        )\n        if found and conf > 0.5:"
new2 = "        found, conf, pos = self._template_match(\n            image, 'play_button.png', play_region, threshold=0.35\n        )\n        if found and conf > 0.35:"

if old2 in text:
    text = text.replace(old2, new2)
    print('OK: play_button threshold atualizado para 0.35')
else:
    print('ERRO: play_button threshold nao encontrado')

p.write_text(text, encoding='utf-8')
print('Guardado.')
