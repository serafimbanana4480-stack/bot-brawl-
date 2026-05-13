"""Quick Install Script - Install only essential dependencies for training"""

import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

packages = [
    "stable-baselines3",
    "gymnasium",
    "ultralytics",
]

print("Installing essential training dependencies...")
for pkg in packages:
    try:
        print(f"  Installing {pkg}...")
        install(pkg)
        print(f"  ✓ {pkg} installed")
    except Exception as e:
        print(f"  ✗ Failed to install {pkg}: {e}")

print("\nVerifying installations...")
for pkg in packages:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"  ✓ {pkg} OK")
    except ImportError:
        print(f"  ✗ {pkg} NOT installed")

print("\nDone! You can now run training.")
