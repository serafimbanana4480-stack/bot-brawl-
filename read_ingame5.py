with open('pylaai_real/state_manager.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(1112, 1125):
    print(f'{i+1:4d}  {lines[i]}', end='')
