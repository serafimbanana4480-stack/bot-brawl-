with open('pylaai_real/state_manager.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(1054, 1071):
    print(f'{i+1:4d}  {lines[i]}', end='')
