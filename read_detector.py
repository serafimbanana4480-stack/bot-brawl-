with open('pylaai_real/unified_state_detector.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()
# Print detect method (lines 793-1000 approx)
for i in range(792, 950):
    print(f'{i+1:4d}  {lines[i]}', end='')
