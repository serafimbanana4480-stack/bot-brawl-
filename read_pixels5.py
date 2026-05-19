with open('pylaai_real/unified_state_detector.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(421, 435):
    print(f'{i+1:4d}  {lines[i]}', end='')
