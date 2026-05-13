"""Check installation - works from any directory.

Usage: python _check_install.py
"""
import sys
import os
from pathlib import Path

# Resolve repo root relative to this script
_repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(_repo_root))
os.environ['PYTHONPATH'] = str(_repo_root)

from backend.brawl_bot.verify_installation import InstallationVerifier

v = InstallationVerifier()
v.run_all_checks()

output_path = _repo_root / "backend" / "brawl_bot" / "_check_output.txt"
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    for comp, res in v.results.items():
        f.write(f"{comp}: {res['status']}\n")
    f.write('---WARNINGS---\n')
    for w in v.warnings:
        f.write(f'WARN: {w["component"]}: {w["issue"]}\n')
    f.write('---ISSUES---\n')
    for i in v.issues:
        f.write(f'ISSUE: {i["component"]}: {i["issue"]}\n')

print('done')
