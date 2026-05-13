"""
installation_script.py

Automated installation script for Brawl Stars bot components.
Handles BlueStacks detection, PylaAI validation, ADB connection, YOLO model downloads,
and Python dependency installation with progress indicators and rollback capability.
"""

import os
import sys
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
import json
import time
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('installation.log')
    ]
)
logger = logging.getLogger(__name__)


class InstallationProgress:
    """Track installation progress"""
    
    def __init__(self):
        self.steps = []
        self.current_step = None
        self.total_steps = 0
        self.completed_steps = 0
    
    def add_step(self, step_name: str):
        self.steps.append(step_name)
        self.total_steps = len(self.steps)
    
    def start_step(self, step_name: str):
        self.current_step = step_name
        logger.info(f"[{self.completed_steps + 1}/{self.total_steps}] Starting: {step_name}")
    
    def complete_step(self, step_name: str):
        self.completed_steps += 1
        logger.info(f"[{self.completed_steps}/{self.total_steps}] Completed: {step_name}")
    
    def get_progress(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return (self.completed_steps / self.total_steps) * 100


class InstallationReport:
    """Generate installation report"""
    
    def __init__(self):
        self.components = {}
        self.errors = []
        self.warnings = []
        self.success = True
        self.start_time = time.time()
        self.end_time = None
    
    def add_component(self, name: str, status: str, details: str = ""):
        self.components[name] = {
            "status": status,
            "details": details,
            "timestamp": time.time()
        }
    
    def add_error(self, error: str):
        self.errors.append(error)
        self.success = False
    
    def add_warning(self, warning: str):
        self.warnings.append(warning)
    
    def generate(self) -> str:
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        report = {
            "success": self.success,
            "duration_seconds": duration,
            "components": self.components,
            "errors": self.errors,
            "warnings": self.warnings
        }
        
        return json.dumps(report, indent=2)


def detect_bluestacks() -> Dict:
    """
    Check if BlueStacks is installed at default paths.
    
    Returns:
        Dict with 'found', 'path', 'version' keys
    """
    logger.info("Detecting BlueStacks installation...")
    
    # Default BlueStacks installation paths on Windows
    default_paths = [
        Path(r"C:\Program Files\BlueStacks_nxt"),
        Path(r"C:\Program Files (x86)\BlueStacks_nxt"),
        Path(r"C:\Program Files\BlueStacks"),
        Path(r"C:\Program Files (x86)\BlueStacks"),
    ]
    
    for path in default_paths:
        if path.exists():
            logger.info(f"BlueStacks found at: {path}")
            
            # Try to get version from HD-Player.exe
            hd_player = path / "HD-Player.exe"
            if hd_player.exists():
                return {
                    "found": True,
                    "path": str(path),
                    "executable": str(hd_player),
                    "version": "detected"
                }
            else:
                return {
                    "found": True,
                    "path": str(path),
                    "executable": None,
                    "version": "unknown"
                }
    
    logger.warning("BlueStacks not found in default paths")
    return {
        "found": False,
        "path": None,
        "executable": None,
        "version": None
    }


def validate_pylai() -> Dict:
    """
    Verify C:\PylaAI exists and is accessible.
    
    Returns:
        Dict with 'valid', 'path', 'accessible' keys
    """
    logger.info("Validating PylaAI installation...")
    
    pylai_path = Path(r"C:\PylaAI")
    
    if not pylai_path.exists():
        logger.error("PylaAI not found at C:\\PylaAI")
        return {
            "valid": False,
            "path": str(pylai_path),
            "accessible": False,
            "error": "Directory does not exist"
        }
    
    # Check if it's accessible (can read/write)
    try:
        test_file = pylai_path / ".access_test"
        test_file.touch()
        test_file.unlink()
        
        logger.info("PylaAI is accessible")
        return {
            "valid": True,
            "path": str(pylai_path),
            "accessible": True
        }
    except Exception as e:
        logger.error(f"PylaAI not accessible: {e}")
        return {
            "valid": False,
            "path": str(pylai_path),
            "accessible": False,
            "error": str(e)
        }


def test_adb_connection() -> Dict:
    """
    Verify ADB can connect to BlueStacks.
    
    Returns:
        Dict with 'connected', 'device', 'error' keys
    """
    logger.info("Testing ADB connection to BlueStacks...")
    
    try:
        # Check if adb is available
        result = subprocess.run(
            ["adb", "version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode != 0:
            return {
                "connected": False,
                "device": None,
                "error": "ADB not found or not in PATH"
            }
        
        # Try to connect to BlueStacks (default port 5555)
        result = subprocess.run(
            ["adb", "connect", "127.0.0.1:5555"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Check connected devices
        result = subprocess.run(
            ["adb", "devices"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if "127.0.0.1:5555" in result.stdout:
            logger.info("ADB connected to BlueStacks")
            return {
                "connected": True,
                "device": "127.0.0.1:5555",
                "error": None
            }
        else:
            logger.warning("ADB not connected to BlueStacks")
            return {
                "connected": False,
                "device": None,
                "error": "BlueStacks device not found"
            }
            
    except subprocess.TimeoutExpired:
        return {
            "connected": False,
            "device": None,
            "error": "ADB command timed out"
        }
    except Exception as e:
        logger.error(f"ADB connection error: {e}")
        return {
            "connected": False,
            "device": None,
            "error": str(e)
        }


def download_yolo_models(
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> Dict:
    """
    Download YOLO models using model_downloader.py.
    
    Args:
        progress_callback: Function callback for progress (bytes_downloaded, total_bytes)
    
    Returns:
        Dict with 'success', 'models', 'errors' keys
    """
    logger.info("Downloading YOLO models...")
    
    try:
        from .model_downloader import get_model_downloader
        
        downloader = get_model_downloader()
        models_to_download = ["yolov8n", "yolov8s", "yolov8m"]
        
        results = {}
        errors = []
        
        for model_key in models_to_download:
            logger.info(f"Downloading {model_key}...")
            
            def callback(downloaded, total):
                if progress_callback:
                    progress_callback(downloaded, total)
            
            result = downloader.download_model(model_key, progress_callback=callback)
            
            if result.get("success"):
                results[model_key] = {
                    "success": True,
                    "path": result.get("path"),
                    "size_mb": result.get("size_mb")
                }
                logger.info(f"{model_key} downloaded successfully")
            else:
                errors.append({
                    "model": model_key,
                    "error": result.get("error")
                })
                logger.error(f"Failed to download {model_key}: {result.get('error')}")
        
        return {
            "success": len(errors) == 0,
            "models": results,
            "errors": errors
        }
        
    except Exception as e:
        logger.error(f"Error downloading YOLO models: {e}")
        return {
            "success": False,
            "models": {},
            "errors": [str(e)]
        }


def install_python_dependencies() -> Dict:
    """
    Install Python dependencies for Brawl Stars components.
    
    Returns:
        Dict with 'success', 'installed', 'errors' keys
    """
    logger.info("Installing Python dependencies...")
    
    dependencies = [
        "pyautogui",
        "ultralytics",
        "numpy",
        "opencv-python",
        "pillow",
        "requests"
    ]
    
    installed = {}
    errors = []
    
    for dep in dependencies:
        logger.info(f"Installing {dep}...")
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", dep],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            if result.returncode == 0:
                installed[dep] = True
                logger.info(f"{dep} installed successfully")
            else:
                error_msg = result.stderr or result.stdout
                errors.append({
                    "dependency": dep,
                    "error": error_msg
                })
                logger.error(f"Failed to install {dep}: {error_msg}")
                
        except subprocess.TimeoutExpired:
            errors.append({
                "dependency": dep,
                "error": "Installation timed out"
            })
            logger.error(f"Timeout installing {dep}")
        except Exception as e:
            errors.append({
                "dependency": dep,
                "error": str(e)
            })
            logger.error(f"Error installing {dep}: {e}")
    
    return {
        "success": len(errors) == 0,
        "installed": installed,
        "errors": errors
    }


def validate_step(step_name: str, validation_func: Callable) -> Dict:
    """
    Validate a specific installation step.
    
    Args:
        step_name: Name of the step to validate
        validation_func: Function to run validation
    
    Returns:
        Dict with 'valid', 'details' keys
    """
    logger.info(f"Validating step: {step_name}")
    
    try:
        result = validation_func()
        logger.info(f"Validation result: {result}")
        return result
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return {
            "valid": False,
            "error": str(e)
        }


def rollback_installation(
    completed_steps: List[str],
    report: InstallationReport
) -> bool:
    """
    Undo installation steps on failure.
    
    Args:
        completed_steps: List of completed step names
        report: Installation report to update
    
    Returns:
        bool indicating success
    """
    logger.info("Rolling back installation...")
    
    for step in reversed(completed_steps):
        logger.info(f"Rolling back: {step}")
        
        if step == "download_yolo_models":
            # Remove downloaded models
            try:
                models_dir = Path(__file__).parent / "models"
                if models_dir.exists():
                    shutil.rmtree(models_dir)
                    logger.info("Removed YOLO models")
            except Exception as e:
                logger.error(f"Failed to remove models: {e}")
        
        elif step == "install_python_dependencies":
            # Cannot easily rollback pip installs
            logger.warning("Cannot rollback pip installations")
        
        # Other steps may not need rollback
    
    logger.info("Rollback complete")
    return True


def generate_troubleshooting_guide() -> str:
    """
    Generate help text for common failures.
    
    Returns:
        str with troubleshooting guide
    """
    guide = """
# Brawl Stars Bot Installation Troubleshooting Guide

## BlueStacks Not Found
- Download BlueStacks from https://www.bluestacks.com/
- Install to default location (C:\\Program Files\\BlueStacks_nxt)
- Start BlueStacks before running installation

## PylaAI Not Found
- PylaAI must be manually installed to C:\\PylaAI
- Download from official PylaAI repository
- Ensure directory is accessible (not read-only)

## ADB Connection Failed
- Enable ADB in BlueStacks: Settings > Advanced > Enable Android Debug Bridge
- Ensure ADB is installed and in PATH
- Try connecting manually: adb connect 127.0.0.1:5555

## YOLO Model Download Failed
- Check internet connection
- Verify Ultralytics GitHub is accessible
- Try manual download from: https://github.com/ultralytics/assets/releases/download/v0.0.0/

## Python Dependencies Failed
- Ensure Python 3.11+ is installed
- Try: python -m pip install --upgrade pip
- Run as administrator if permission errors occur
- Check firewall/antivirus is blocking downloads

## General Issues
- Check installation.log for detailed error messages
- Ensure all prerequisites are met before running
- Run installation as administrator
- Disable antivirus temporarily if blocking installation
"""
    return guide


def generate_installation_report(report: InstallationReport) -> str:
    """
    Generate summary of installed components.
    
    Args:
        report: Installation report
    
    Returns:
        str with installation summary
    """
    summary = report.generate()
    logger.info(f"Installation report:\n{summary}")
    return summary


def main():
    """Main installation function"""
    
    logger.info("=" * 60)
    logger.info("Brawl Stars Bot Installation Script")
    logger.info("=" * 60)
    
    # Initialize tracking
    progress = InstallationProgress()
    report = InstallationReport()
    completed_steps = []
    
    # Define installation steps
    steps = [
        ("detect_bluestacks", detect_bluestacks),
        ("validate_pylai", validate_pylai),
        ("test_adb_connection", test_adb_connection),
        ("download_yolo_models", lambda: download_yolo_models()),
        ("install_python_dependencies", install_python_dependencies)
    ]
    
    for step_name, step_func in steps:
        progress.add_step(step_name)
    
    # Execute installation steps
    for step_name, step_func in steps:
        progress.start_step(step_name)
        
        try:
            result = step_func()
            
            # Validate result
            if isinstance(result, dict):
                if result.get("success") or result.get("found") or result.get("valid") or result.get("connected"):
                    report.add_component(step_name, "success", str(result))
                    completed_steps.append(step_name)
                    progress.complete_step(step_name)
                else:
                    report.add_component(step_name, "failed", str(result))
                    report.add_error(f"{step_name} failed: {result.get('error', 'Unknown error')}")
                    
                    # Rollback on failure
                    rollback_installation(completed_steps, report)
                    
                    # Generate troubleshooting guide
                    guide = generate_troubleshooting_guide()
                    logger.warning(f"\nTroubleshooting Guide:\n{guide}")
                    
                    # Generate report
                    generate_installation_report(report)
                    
                    logger.error("Installation failed. See troubleshooting guide above.")
                    sys.exit(1)
            else:
                report.add_component(step_name, "unknown", str(result))
                
        except Exception as e:
            logger.error(f"Error executing {step_name}: {e}")
            report.add_component(step_name, "error", str(e))
            report.add_error(f"{step_name} error: {str(e)}")
            
            # Rollback on failure
            rollback_installation(completed_steps, report)
            
            # Generate troubleshooting guide
            guide = generate_troubleshooting_guide()
            logger.warning(f"\nTroubleshooting Guide:\n{guide}")
            
            # Generate report
            generate_installation_report(report)
            
            logger.error("Installation failed. See troubleshooting guide above.")
            sys.exit(1)
    
    # Installation complete
    logger.info("=" * 60)
    logger.info("Installation completed successfully!")
    logger.info("=" * 60)
    
    # Generate final report
    final_report = generate_installation_report(report)
    
    # Save report to file
    with open("installation_report.json", "w") as f:
        f.write(final_report)
    
    logger.info("Installation report saved to installation_report.json")
    logger.info("Installation log saved to installation.log")
    
    return report


if __name__ == "__main__":
    main()
