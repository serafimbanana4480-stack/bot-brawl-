with open('pylaai_real/unified_state_detector.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Ver linhas 430-475
for i in range(430, 475):
    print(str(i+1).rjust(4) + '  ' + lines[i].rstrip())
