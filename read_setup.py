with open('wrapper.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for i in range(1128, 1200):
    print(f'{i+1:4d}  {lines[i]}', end='')
