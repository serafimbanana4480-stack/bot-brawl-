"""
verify_installation.py

Comprehensive installation verification script for Brawl Stars bot.
Checks all dependencies and provides fix instructions.
"""

import sys
import os
import subprocess
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Add current directory to sys.path to allow imports when run as script
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

# Import emulator detector utilities
try:
    from emulator_detector import get_emulator_detector, get_adb_path
except Exception as e:
    print(f"DEBUG: Initial import failed: {e}")
    # Support running as a script from repository root where package imports may differ
    try:
        from backend.brawl_bot.emulator_detector import get_emulator_detector, get_adb_path
    except Exception as e2:
        print(f"DEBUG: Secondary import failed: {e2}")
        # Defer import errors until runtime checks that need them
        get_emulator_detector = None  # type: ignore
        get_adb_path = None  # type: ignore


class InstallationVerifier:
    """Verifies all Brawl Stars bot dependencies"""
    
    def __init__(self):
        self.issues: List[Dict] = []
        self.warnings: List[Dict] = []
        self.results: Dict = {}
        self.adb_diagnostics: Dict = {}
        self.detector_raw = None
    
    def check_python_version(self) -> bool:
        """Check Python 3.11+ is installed"""
        version = sys.version_info
        if version.major >= 3 and version.minor >= 11:
            self.results["python"] = {
                "status": "ok",
                "version": f"{version.major}.{version.minor}.{version.micro}"
            }
            return True
        else:
            self.issues.append({
                "component": "Python",
                "issue": f"Python {version.major}.{version.minor} installed, need 3.11+",
                "fix": "Install Python 3.11 or higher from python.org"
            })
            self.results["python"] = {
                "status": "error",
                "version": f"{version.major}.{version.minor}.{version.micro}"
            }
            return False
    
    def check_nodejs(self) -> bool:
        """Check Node.js 18+ is installed"""
        try:
            result = subprocess.run(
                ["node", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                version_str = result.stdout.strip().replace("v", "")
                major = int(version_str.split(".")[0])
                if major >= 18:
                    self.results["nodejs"] = {
                        "status": "ok",
                        "version": version_str
                    }
                    return True
                else:
                    self.issues.append({
                        "component": "Node.js",
                        "issue": f"Node.js {version_str} installed, need 18+",
                        "fix": "Install Node.js 18 or higher from nodejs.org"
                    })
                    self.results["nodejs"] = {
                        "status": "error",
                        "version": version_str
                    }
                    return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.issues.append({
                "component": "Node.js",
                "issue": "Node.js not found",
                "fix": "Install Node.js 18 or higher from nodejs.org"
            })
            self.results["nodejs"] = {
                "status": "error",
                "version": "not found"
            }
            return False
    
    def check_ollama(self) -> bool:
        """Check Ollama is running"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.results["ollama"] = {
                    "status": "ok",
                    "message": "Ollama running"
                }
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.warnings.append({
                "component": "Ollama",
                "issue": "Ollama not found or not running",
                "fix": "Install Ollama from ollama.ai and start with: ollama serve"
            })
            self.results["ollama"] = {
                "status": "warning",
                "message": "not found"
            }
            return False
    
    def check_adb(self) -> bool:
        """Check ADB is available (high-level, non-functional)"""
        # Use get_adb_path if available
        adb_path = None
        try:
            if callable(get_adb_path):
                adb_path = get_adb_path()
        except Exception:
            adb_path = None

        # Attempt to run adb --version using returned path or 'adb'
        tried = adb_path or "adb"
        try:
            result = subprocess.run(
                [tried, "version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self.results["adb"] = {
                    "status": "ok",
                    "location": adb_path or "adb",
                    "version_output": result.stdout.strip()
                }
                return True
            else:
                # Nonzero return - treat as error
                self.issues.append({
                    "component": "ADB",
                    "issue": f"ADB returned non-zero exit code when checking version: {result.returncode}",
                    "fix": "Ensure adb is executable and accessible"
                })
                self.results["adb"] = {
                    "status": "error",
                    "location": adb_path or "adb",
                    "version_output": result.stdout.strip(),
                    "version_error": result.stderr.strip()
                }
                return False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            self.issues.append({
                "component": "ADB",
                "issue": "ADB not found or timed out",
                "fix": "Install Android SDK platform-tools or place adb in repository"
            })
            self.results["adb"] = {
                "status": "error",
                "location": adb_path or "adb",
                "version_output": "",
                "version_error": "not found or timeout"
            }
            return False
    
    def perform_adb_diagnostics(self):
        """Run adb devices and adb version capturing stdout/stderr and return structured diagnostics"""
        adb_path = None
        try:
            if callable(get_adb_path):
                adb_path = get_adb_path()
        except Exception:
            adb_path = "adb"

        adb_path = adb_path or "adb"
        diagnostics: Dict = {"adb_path": adb_path}

        # adb version
        try:
            ver = subprocess.run([adb_path, "version"], capture_output=True, text=True, timeout=10)
            diagnostics["version"] = {
                "returncode": ver.returncode,
                "stdout": ver.stdout,
                "stderr": ver.stderr
            }
        except Exception as e:
            diagnostics["version"] = {"error": str(e)}

        # adb devices
        try:
            dev = subprocess.run([adb_path, "devices"], capture_output=True, text=True, timeout=10)
            diagnostics["devices"] = {
                "returncode": dev.returncode,
                "stdout": dev.stdout,
                "stderr": dev.stderr
            }
        except Exception as e:
            diagnostics["devices"] = {"error": str(e)}

        self.adb_diagnostics = diagnostics
        return diagnostics
    
    def check_bluestacks(self) -> bool:
        """Check BlueStacks is installed"""
        common_paths = [
            r"C:\Program Files\BlueStacks_nxt",
            r"C:\Program Files\BlueStacks",
            r"C:\Program Files (x86)\BlueStacks_nxt",
            r"C:\Program Files (x86)\BlueStacks"
        ]
        
        for path in common_paths:
            if Path(path).exists():
                self.results["bluestacks"] = {
                    "status": "ok",
                    "location": path
                }
                return True
        
        self.warnings.append({
            "component": "BlueStacks",
            "issue": "BlueStacks not found in common locations",
            "fix": "Install BlueStacks 5 from bluestacks.com (optional - other emulators supported)"
        })
        self.results["bluestacks"] = {
            "status": "warning",
            "location": "not found"
        }
        return False
    
    def check_pylai(self) -> bool:
        """Check PylaAI exists at C:\\PylaAI"""
        pylai_path = Path(r"C:\PylaAI")
        if pylai_path.exists():
            self.results["pylai"] = {
                "status": "ok",
                "location": str(pylai_path)
            }
            return True
        
        self.warnings.append({
            "component": "PylaAI",
            "issue": "PylaAI not found at C:\\PylaAI",
            "fix": "Install PylaAI to C:\\PylaAI or update install_path in code"
        })
        self.results["pylai"] = {
            "status": "warning",
            "location": "not found"
        }
        return False
    
    def check_yolo_models(self) -> bool:
        """Check YOLO models validity using model_validator"""
        try:
            from model_validator import validate_all_models
        except ImportError:
            try:
                from backend.brawl_bot.model_validator import validate_all_models
            except ImportError:
                self.warnings.append({
                    "component": "YOLO Models",
                    "issue": "model_validator not importable - skipping deep validation",
                    "fix": "Ensure model_validator.py is in the same directory"
                })
                validate_all_models = None

        # Fall back to simple existence check if validator unavailable
        if validate_all_models is None:
            templates = ["thumbs_down.png", "play_button.png", "brawler_select.png", "joystick.png"]
            for t in templates:
                if not (BOT_ROOT / "images" / t).exists():
                    self.warnings.append({
                        "component": "Templates",
                        "issue": f"Template {t} not found",
                        "fix": "Download with model_downloader.py or ultralytics"
                    })
            self.results["templates"] = {
                "status": "warning" if any(not (BOT_ROOT / "images" / t).exists() for t in templates) else "ok",
                "templates": templates
            }
            model_paths = [
                BOT_ROOT / "models" / "yolov8n.pt",
                BOT_ROOT / "models" / "yolov8m.pt"
            ]
            found = [p.name for p in model_paths if p.exists()]
            if found:
                self.results["yolo_models"] = {"status": "ok", "models": found, "note": "existence-only check"}
                return True
            self.warnings.append({"component": "YOLO Models", "issue": "No YOLO models found", "fix": "Download with model_downloader.py"})
            self.results["yolo_models"] = {"status": "warning", "models": []}
            return False

        validation = validate_all_models(delete_fakes=False)
        valid_models = validation.get("valid", [])
        fake_models = validation.get("fake", [])
        invalid_models = validation.get("invalid", [])

        model_names = [m["name"] for m in valid_models]

        if valid_models:
            self.results["yolo_models"] = {
                "status": "ok",
                "models": model_names,
                "valid_count": len(valid_models),
                "fake_count": len(fake_models),
                "invalid_count": len(invalid_models)
            }
        elif fake_models or invalid_models:
            self.warnings.append({
                "component": "YOLO Models",
                "issue": f"No valid models. {len(fake_models)} fake, {len(invalid_models)} invalid.",
                "fix": "Train a Brawl Stars model or download a valid one"
            })
            self.results["yolo_models"] = {
                "status": "warning",
                "models": [m["name"] for m in fake_models + invalid_models],
                "fake": [m["name"] for m in fake_models],
                "invalid": [m["name"] for m in invalid_models]
            }
        else:
            self.warnings.append({
                "component": "YOLO Models",
                "issue": "No YOLO models found",
                "fix": "Download with model_downloader.py or ultralytics"
            })
            self.results["yolo_models"] = {"status": "warning", "models": []}
            return False

        # Log fake models explicitly
        for m in fake_models:
            self.warnings.append({
                "component": "YOLO Models",
                "issue": f"FAKE model: {m['name']} — {m.get('reason', 'unknown')}",
                "fix": "Remove or replace with a real Brawl Stars model"
            })

        return bool(valid_models)
    
    def check_python_packages(self) -> bool:
        """Check required Python packages"""
        required_packages = [
            ("fastapi", "fastapi"),
            ("uvicorn", "uvicorn"),
            ("pyautogui", "pyautogui"),
            ("ultralytics", "ultralytics"),
            ("numpy", "numpy"),
            ("opencv-python", "cv2"),
            ("psutil", "psutil"),
            ("pywin32", "win32api")
        ]
        
        missing_packages = []
        found_packages = []
        for package_name, import_name in required_packages:
            try:
                __import__(import_name)
                found_packages.append(package_name)
            except ImportError:
                missing_packages.append(package_name)
        
        if missing_packages:
            self.issues.append({
                "component": "Python Packages",
                "issue": f"Missing packages: {', '.join(missing_packages)}",
                "fix": f"Install with: pip install {' '.join(missing_packages)}"
            })
            self.results["python_packages"] = {
                "status": "error",
                "missing": missing_packages,
                "found": found_packages
            }
            return False
        
        self.results["python_packages"] = {
            "status": "ok",
            "packages": found_packages
        }
        return True
    
    def check_shared_module(self) -> bool:
        """Check shared module is accessible"""
        # Note: shared.auth module was removed due to being non-functional
        # Authentication is handled by backend/security.py with JWT validation
        self.results["shared_module"] = {
            "status": "skipped",
            "message": "shared.auth module removed - using backend/security.py instead"
        }
        return True
    
    def check_backend_startup(self) -> bool:
        """Check backend can start (basic import test)"""
        try:
            # File is in backend/brawl_bot, so go up 2 levels to reach soberana-omega
            soberana_omega_dir = Path(__file__).parent.parent.parent
            sys.path.insert(0, str(soberana_omega_dir))
            
            # Import with proper path
            import importlib.util
            api_path = soberana_omega_dir / "backend" / "interface" / "api.py"
            
            spec = importlib.util.spec_from_file_location("api", api_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                self.results["backend"] = {
                    "status": "ok",
                    "message": "Backend imports successfully"
                }
                return True
            else:
                raise ImportError("Could not load api.py")
        except Exception as e:
            self.issues.append({
                "component": "Backend",
                "issue": f"Cannot import backend: {e}",
                "fix": "Fix import errors in backend code or shared module paths"
            })
            self.results["backend"] = {
                "status": "error",
                "error": str(e)
            }
            return False
    
    def check_frontend_build(self) -> bool:
        """Check frontend can build (basic package.json check)"""
        frontend_dir = Path(__file__).parent.parent.parent / "frontend"
        package_json = frontend_dir / "package.json"
        
        if package_json.exists():
            self.results["frontend"] = {
                "status": "ok",
                "location": str(frontend_dir)
            }
            return True
        
        self.warnings.append({
            "component": "Frontend",
            "issue": "Frontend package.json not found",
            "fix": "Ensure frontend directory exists with package.json"
        })
        self.results["frontend"] = {
            "status": "warning",
            "location": "not found"
        }
        return False
    
    def run_all_checks(self) -> Dict:
        """Run all verification checks"""
        print("=" * 60)
        print("BRAWL STARS BOT INSTALLATION VERIFICATION")
        print("=" * 60)
        print()
        
        self.check_python_version()
        self.check_nodejs()
        self.check_ollama()
        self.check_adb()
        self.check_bluestacks()
        self.check_pylai()
        self.check_yolo_models()
        self.check_python_packages()
        self.check_shared_module()
        self.check_backend_startup()
        self.check_frontend_build()
        
        return self.results
    
    def print_report(self):
        """Print verification report"""
        print("\n" + "=" * 60)
        print("VERIFICATION RESULTS")
        print("=" * 60)
        print()
        
        for component, result in self.results.items():
            status = result["status"].upper()
            print(f"{component:20} {status:10}")
            if "version" in result:
                print(f"{'':20} Version: {result['version']}")
            if "location" in result:
                print(f"{'':20} Location: {result['location']}")
            print()
        
        if self.issues:
            print("=" * 60)
            print(f"CRITICAL ISSUES ({len(self.issues)})")
            print("=" * 60)
            print()
            for issue in self.issues:
                print(f"❌ {issue['component']}: {issue['issue']}")
                print(f"   Fix: {issue['fix']}")
                print()
        
        if self.warnings:
            print("=" * 60)
            print(f"WARNINGS ({len(self.warnings)})")
            print("=" * 60)
            print()
            for warning in self.warnings:
                print(f"⚠️  {warning['component']}: {warning['issue']}")
                print(f"   Fix: {warning['fix']}")
                print()
        
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total checks: {len(self.results)}")
        print(f"OK: {sum(1 for r in self.results.values() if r['status'] == 'ok')}")
        print(f"Warnings: {sum(1 for r in self.results.values() if r['status'] == 'warning')}")
        print(f"Errors: {sum(1 for r in self.results.values() if r['status'] == 'error')}")
        print()
        
        if self.issues:
            print("❌ CRITICAL ISSUES FOUND - Fix before running bot")
        elif self.warnings:
            print("⚠️  WARNINGS FOUND - Bot may have limited functionality")
        else:
            print("✅ ALL CHECKS PASSED - Ready to run")
    
    def save_report(self, filename: str = None):
        """Save verification report to file"""
        if filename is None:
            filename = str(Path(__file__).parent / "verify_installation_report.json")

        report = {
            "results": self.results,
            "issues": self.issues,
            "warnings": self.warnings,
            "adb_diagnostics": self.adb_diagnostics,
            "detector": self.detector_raw,
            "summary": {
                "total": len(self.results),
                "ok": sum(1 for r in self.results.values() if r['status'] == 'ok'),
                "warnings": sum(1 for r in self.results.values() if r['status'] == 'warning'),
                "errors": sum(1 for r in self.results.values() if r['status'] == 'error')
            }
        }
        
        # Ensure parent directory exists
        out_path = Path(filename)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\nReport saved to: {filename}")


if __name__ == "__main__":
    verifier = InstallationVerifier()
    results = verifier.run_all_checks()

    # Run emulator detector if available
    detector_data = None
    try:
        if callable(get_emulator_detector):
            detector = get_emulator_detector()
            emulators = detector.detect_all()
            # Serialize dataclasses to plain dicts
            detector_data = [e.__dict__ for e in emulators]
            verifier.detector_raw = detector_data
            # incorporate into results
            verifier.results["emulator_detector"] = {
                "status": "ok" if detector_data else "warning",
                "count": len(detector_data) if detector_data is not None else 0
            }
        else:
            verifier.results["emulator_detector"] = {
                "status": "warning",
                "message": "emulator_detector import not available"
            }
    except Exception as e:
        verifier.results["emulator_detector"] = {
            "status": "error",
            "error": str(e)
        }
        verifier.issues.append({
            "component": "Emulator Detector",
            "issue": f"Error running detector.detect_all(): {e}",
            "fix": "Ensure emulator_detector.py is importable and functioning"
        })

    # Perform adb diagnostics (version + devices)
    try:
        adb_diag = verifier.perform_adb_diagnostics()
        verifier.results.setdefault("adb", {}).update({"diagnostics_included": True})
    except Exception as e:
        adb_diag = {"error": str(e)}
        verifier.adb_diagnostics = adb_diag

    verifier.print_report()

    # Save report to expected path
    out_file = str(Path(__file__).parent / "verify_installation_report.json")
    verifier.save_report(out_file)

    # Determine exit status: non-zero if python packages missing or zero emulators detected
    exit_code = 0
    python_pkg_status = verifier.results.get("python_packages", {}).get("status")
    if python_pkg_status == "error":
        exit_code = 2
    # adb error
    adb_status = verifier.results.get("adb", {}).get("status")
    if adb_status == "error":
        exit_code = max(exit_code, 3)
    # zero emulators
    em_count = 0
    try:
        em_count = len(verifier.detector_raw) if verifier.detector_raw is not None else 0
    except Exception:
        em_count = 0

    if em_count == 0:
        # treat zero emulators as critical for this verification
        exit_code = max(exit_code, 4)

    if exit_code != 0:
        print(f"Exiting with code {exit_code} due to critical verification failures")
        sys.exit(exit_code)

    print("All critical checks passed")
    sys.exit(0)
