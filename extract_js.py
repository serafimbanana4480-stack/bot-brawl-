import re

with open('pylaai_real/dashboard_server.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Find the HTML content
m = re.search(r"_DASHBOARD_HTML = '''(.+?)'''\n\n# -+", content, re.DOTALL)
if not m:
    print("Could not find HTML")
    exit(1)

html = m.group(1)

# Find the script content
sm = re.search(r'<script>(.+?)</script>', html, re.DOTALL)
if not sm:
    print("Could not find script")
    exit(1)

js = sm.group(1)

with open('dash_debug.js', 'w', encoding='utf-8') as f:
    f.write(js)

print(f"JS lines: {len(js.split(chr(10)))}")
print(f"JS chars: {len(js)}")

# Check for the error pattern
if "split('\n" in js:
    idx = js.find("split('")
    print(f"Found split at: {repr(js[idx:idx+20])}")
