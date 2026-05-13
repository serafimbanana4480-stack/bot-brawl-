"""
Enterprise AI Platform - Complete Setup Script
Run this to install all dependencies and verify the system
"""

import subprocess
import sys
import os


def run_command(cmd, description):
    print(f"\n{'='*60}")
    print(f"[SETUP] {description}")
    print(f"{'='*60}")
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] {description} failed:")
        print(e.stderr)
        return False


def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║      ENTERPRISE AI MULTI-AGENT PLATFORM v2.0           ║
    ║                                                           ║
    ║      Complete Setup & Installation                         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)

    commands = [
        ("python --version", "Checking Python version"),
        ("pip install --upgrade pip", "Upgrading pip"),
    ]

    for cmd, desc in commands:
        run_command(cmd, desc)

    print("\n" + "="*60)
    print("[SETUP] Installing Core Dependencies")
    print("="*60)

    core_packages = [
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "pandas>=2.0.0",
    ]

    for package in core_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("\n" + "="*60)
    print("[SETUP] Installing AI/ML Dependencies")
    print("="*60)

    ai_packages = [
        "torch>=2.0.0",
        "torchvision>=0.15.0",
    ]

    for package in ai_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("\n" + "="*60)
    print("[SETUP] Installing Vision Dependencies")
    print("="*60)

    vision_packages = [
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=10.0.0",
        "scikit-image>=0.21.0",
    ]

    for package in vision_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("\n" + "="*60)
    print("[SETUP] Installing RL Dependencies")
    print("="*60)

    rl_packages = [
        "stable-baselines3>=2.0.0",
        "gymnasium>=0.29.0",
    ]

    for package in rl_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("\n" + "="*60)
    print("[SETUP] Installing API Dependencies")
    print("="*60)

    api_packages = [
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "websockets>=12.0",
        "pydantic>=2.0.0",
    ]

    for package in api_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("\n" + "="*60)
    print("[SETUP] Installing Infrastructure Dependencies")
    print("="*60)

    infra_packages = [
        "redis>=5.0.0",
        "sentence-transformers>=2.2.0",
        "psutil>=5.9.0",
        "python-dotenv>=1.0.0",
        "pyyaml>=6.0",
    ]

    for package in infra_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("\n" + "="*60)
    print("[SETUP] Installing Development Tools")
    print("="*60)

    dev_packages = [
        "pytest>=7.4.0",
        "pytest-asyncio>=0.23.0",
        "httpx>=0.26.0",
    ]

    for package in dev_packages:
        run_command(f"pip install {package}", f"Installing {package}")

    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║      INSTALLATION COMPLETED!                              ║
    ║                                                           ║
    ║      Next steps:                                          ║
    ║      1. Run: python enterprise/setup.py                    ║
    ║      2. Run: python enterprise/main.py                   ║
    ║      3. Run: cd enterprise/dashboard && npm install       ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    main()
