"""
emulator_detector.py

Sistema de detecção automática de emuladores Android.
Suporta detecção via ADB e window detection para múltiplos emuladores.
"""

import subprocess
import logging
import os
import sys
from typing import List, Dict, Optional
from dataclasses import dataclass
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# ADB path configuration
def get_adb_path() -> str:
    """Returns the ADB executable path, preferring project-bundled binaries, then system PATH, then Android SDK.

    This function deterministically logs which path was chosen and why. It prefers the repository's adb (adb or adb.exe)
    in the repository root or platform-tools directory to avoid adb client/server mismatches as per Decision D002.
    """
    bundle_dir = Path(__file__).parent.parent.parent

    # Determine platform-specific executable name
    adb_names = ["adb.exe", "adb"] if sys.platform.startswith("win") else ["adb"]

    # 1) Prefer ./adb or ./adb.exe in repository root (explicit bundled adb)
    for name in adb_names:
        candidate = bundle_dir / name
        if candidate.exists() and candidate.stat().st_size > 0:
            chosen = str(candidate)
            logger.info(f"Selected ADB path: {chosen} (reason: repository-bundled executable)")
            return chosen

    # 2) Prefer platform-tools/ad b in repository (repo/platform-tools/adb or adb.exe)
    for name in adb_names:
        candidate = bundle_dir / "platform-tools" / name
        if candidate.exists() and candidate.stat().st_size > 0:
            chosen = str(candidate)
            logger.info(f"Selected ADB path: {chosen} (reason: repository platform-tools)")
            return chosen

    # 3) Try system PATH (adb)
    try:
        result = subprocess.run(["adb", "version"], capture_output=True, timeout=5)
        if result.returncode == 0:
            logger.info("Selected ADB path: adb (reason: system PATH adb detected)")
            return "adb"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        logger.debug("System adb not found or timed out when checking 'adb version'")

    # 4) Try Android SDK platform-tools via ANDROID_HOME or ANDROID_SDK_ROOT
    android_home = os.environ.get("ANDROID_HOME") or os.environ.get("ANDROID_SDK_ROOT")
    if android_home:
        for name in adb_names:
            candidate = Path(android_home) / "platform-tools" / name
            if candidate.exists() and candidate.stat().st_size > 0:
                chosen = str(candidate)
                logger.info(f"Selected ADB path: {chosen} (reason: ANDROID_HOME/SDK platform-tools)")
                return chosen

    # Nothing found; warn and return 'adb' as a last-resort to keep existing behavior (error will be raised when used)
    logger.warning("ADB not found in repository, system PATH, or Android SDK; falling back to 'adb' (may fail)")
    return "adb"


@dataclass
class EmulatorInfo:
    """Informações sobre um emulador detectado"""
    name: str
    type: str  # "bluestacks", "ldplayer", "nox", "memu", "unknown"
    adb_id: Optional[str] = None
    window_title: Optional[str] = None
    window_handle: Optional[int] = None
    connected: bool = False


class EmulatorDetector:
    """Detetor de emuladores Android"""
    
    # Padrões de window title para diferentes emuladores
    WINDOW_PATTERNS = {
        "bluestacks": [
            r"BlueStacks",
            r"BlueStacks App Player",
            r"HD-Player",
            r"HD-Player\.exe",
            r"BlueStacksApp",
            r"Bstk"
        ],
        "ldplayer": [
            r"LDPlayer",
            r"LD\d+",
            r"Leidian"
        ],
        "nox": [
            r"NoxPlayer",
            r"Nox App Player",
            r"Nox"
        ],
        "memu": [
            r"MEMu",
            r"Microvirt"
        ]
    }
    
    def __init__(self):
        self.available_emulators: List[EmulatorInfo] = []
        
    def detect_adb_devices(self) -> List[EmulatorInfo]:
        """Detecta emuladores via ADB"""
        emulators = []
        
        try:
            # First, try BlueStacks ADB if BlueStacks is detected
            bluestacks_adb = self._get_bluestacks_adb()
            if bluestacks_adb:
                logger.info(f"BlueStacks ADB detected at: {bluestacks_adb}")
                logger.debug(f"Attempting device detection with BlueStacks ADB")
                bluestacks_emulators = self._try_adb_with_path(bluestacks_adb, "bluestacks")
                logger.info(f"BlueStacks ADB detected {len(bluestacks_emulators)} emulator(s)")
                emulators.extend(bluestacks_emulators)
            else:
                logger.info("BlueStacks ADB not found at standard paths")
            
            # Then try standard ADB
            adb_path = get_adb_path()
            logger.info(f"Using standard ADB path: {adb_path}")
            logger.debug(f"ADB path selection reason: checked in order: repository root, platform-tools, system PATH, Android SDK")
            
            # Executar adb devices
            logger.debug(f"Executing ADB command: {adb_path} devices")
            result = subprocess.run(
                [adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            logger.debug(f"ADB command return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"ADB stdout (first 500 chars): {result.stdout[:500]}")
            if result.stderr:
                logger.debug(f"ADB stderr (first 500 chars): {result.stderr[:500]}")
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Pular header
                logger.debug(f"ADB output parsed: {len(lines)} device lines found")
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        status = parts[1]
                        
                        logger.debug(f"Processing device: {device_id} with status: {status}")
                        
                        if status == "device":
                            # Tentar identificar o tipo de emulador
                            emulator_type = self._identify_emulator_from_adb(device_id)
                            logger.debug(f"Identified emulator type: {emulator_type} for device: {device_id}")
                            
                            # Avoid duplicates
                            if not any(e.adb_id == device_id for e in emulators):
                                emulators.append(EmulatorInfo(
                                    name=device_id,
                                    type=emulator_type,
                                    adb_id=device_id,
                                    connected=True
                                ))
                                
                                logger.info(f"Emulator ADB detected: {device_id} (type: {emulator_type}, status: connected)")
                            else:
                                logger.debug(f"Skipping duplicate device: {device_id}")
                        else:
                            logger.debug(f"Device {device_id} has status: {status} (not connected)")
                logger.info(f"Total emulators detected via ADB: {len(emulators)}")
            else:
                logger.warning(f"ADB command failed with return code: {result.returncode}")
                logger.warning(f"ADB stderr: {result.stderr}")
                            
        except FileNotFoundError as e:
            logger.error(f"ADB not found at {get_adb_path()}", extra={"context": {"error_type": "FileNotFoundError", "adb_path": str(get_adb_path())}})
            logger.info("Install Android SDK platform-tools or ensure adb.exe is in soberana-omega directory")
        except subprocess.TimeoutExpired:
            logger.error("Timeout ao executar adb devices", extra={"context": {"error_type": "TimeoutExpired", "timeout_seconds": 10}})
        except Exception as e:
            logger.error(f"Erro ao detectar dispositivos ADB: {e}", extra={"context": {"error_type": type(e).__name__, "error_message": str(e)}})
        
        return emulators
    
    def _get_bluestacks_adb(self) -> Optional[str]:
        """Get BlueStacks HD-Adb.exe path if BlueStacks is installed"""
        bluestacks_paths = [
            r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe",
            r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe",
            r"C:\Program Files\BlueStacks\HD-Adb.exe",
        ]
        
        for path in bluestacks_paths:
            if Path(path).exists():
                return path
        return None
    
    def _try_adb_with_path(self, adb_path: str, emulator_type: str) -> List[EmulatorInfo]:
        """Try to detect devices using a specific ADB path"""
        emulators = []
        logger.info(f"Attempting device detection with ADB path: {adb_path} for emulator type: {emulator_type}")
        
        try:
            logger.debug(f"Executing command: {adb_path} devices")
            result = subprocess.run(
                [adb_path, "devices"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            logger.debug(f"Command return code: {result.returncode}")
            if result.stdout:
                logger.debug(f"Command stdout (first 500 chars): {result.stdout[:500]}")
            if result.stderr:
                logger.debug(f"Command stderr (first 500 chars): {result.stderr[:500]}")
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]
                logger.debug(f"Parsed {len(lines)} device lines from output")
                
                for line in lines:
                    if not line.strip():
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 2:
                        device_id = parts[0]
                        status = parts[1]
                        
                        logger.debug(f"Processing device: {device_id} with status: {status}")
                        
                        if status == "device":
                            emulators.append(EmulatorInfo(
                                name=device_id,
                                type=emulator_type,
                                adb_id=device_id,
                                connected=True
                            ))
                            
                            logger.info(f"Emulator {emulator_type} detected via ADB: {device_id} (status: connected)")
                        else:
                            logger.debug(f"Device {device_id} has status: {status} (not connected)")
                
                logger.info(f"Total {emulator_type} emulators detected: {len(emulators)}")
            else:
                logger.warning(f"ADB command failed with return code: {result.returncode}")
                logger.warning(f"Command stderr: {result.stderr}")
                                
        except subprocess.TimeoutExpired:
            logger.error(f"Timeout executing ADB command with path: {adb_path}", extra={"context": {"error_type": "TimeoutExpired", "adb_path": adb_path, "emulator_type": emulator_type, "timeout_seconds": 10}})
        except Exception as e:
            logger.error(f"Failed to detect devices with {adb_path}: {e}", extra={"context": {"error_type": type(e).__name__, "error_message": str(e), "adb_path": adb_path, "emulator_type": emulator_type}})
        
        return emulators
    
    def _identify_emulator_from_adb(self, device_id: str) -> str:
        """Identifica tipo de emulador baseado no ID do dispositivo ADB"""
        device_id_lower = device_id.lower()
        logger.debug(f"Identifying emulator type from device ID: {device_id}")
        
        # BlueStacks typically uses localhost:5555
        if "127.0.0.1:5555" in device_id_lower or "localhost:5555" in device_id_lower:
            logger.debug(f"Device ID matches BlueStacks pattern (localhost:5555): {device_id}")
            return "bluestacks"
        elif any(x in device_id_lower for x in ["bluestacks", "bs"]):
            logger.debug(f"Device ID contains BlueStacks keywords: {device_id}")
            return "bluestacks"
        elif any(x in device_id_lower for x in ["ldplayer", "ld"]):
            logger.debug(f"Device ID contains LDPlayer keywords: {device_id}")
            return "ldplayer"
        elif "nox" in device_id_lower:
            logger.debug(f"Device ID contains Nox keywords: {device_id}")
            return "nox"
        elif "memu" in device_id_lower:
            logger.debug(f"Device ID contains MEMU keywords: {device_id}")
            return "memu"
        elif "emulator" in device_id_lower:
            # Generic emulator - try to infer from window detection
            logger.debug(f"Device ID contains generic 'emulator' keyword: {device_id}, will try window detection")
            return "unknown"
        
        logger.debug(f"Device ID does not match any known emulator patterns: {device_id}, defaulting to 'unknown'")
        return "unknown"
    
    def detect_window_emulators(self) -> List[EmulatorInfo]:
        """Detecta emuladores via window detection (Windows only)"""
        logger.info("Starting window-based emulator detection")
        emulators: List[EmulatorInfo] = []

        win32_available = False
        psutil_available = False

        try:
            import win32gui  # type: ignore
            import win32process  # type: ignore
            win32_available = True
            logger.info("pywin32 is available for window detection")
            logger.debug("Imported win32gui and win32process successfully")
        except ImportError as e:
            logger.warning("pywin32 (win32gui/win32process) not available; falling back to process detection", extra={"context": {"error_type": "ImportError", "error_message": str(e)}})

        try:
            import psutil  # type: ignore
            psutil_available = True
            logger.info("psutil is available for process inspection")
            logger.debug("Imported psutil successfully")
        except ImportError as e:
            logger.warning("psutil not available; some detection features may be limited", extra={"context": {"error_type": "ImportError", "error_message": str(e)}})

        if not win32_available:
            logger.info("Using process-based detection because window APIs are not available")
            logger.debug("Window detection unavailable, switching to process-based detection")
            return self._detect_by_process()

        # At this point we have win32gui (pywin32); use EnumWindows
        try:
            logger.debug("Starting window enumeration with EnumWindows")
            # Define the callback locally; EnumWindows will call it for each window
            import win32gui  # type: ignore
            import win32process  # type: ignore
            import psutil as _psutil  # type: ignore

            def enum_windows_callback(hwnd, emulators_list):
                try:
                    if win32gui.IsWindowVisible(hwnd):
                        title = win32gui.GetWindowText(hwnd)
                        logger.debug(f"Checking visible window: {title[:100] if title else 'empty'} (hwnd: {hwnd})")
                        if title:
                            emulator_type = self._identify_emulator_from_title(title)
                            logger.debug(f"Window title '{title[:50]}' identified as: {emulator_type}")
                            
                            # Ignorar janelas de Overlay que costumam cobrir o ecrã todo
                            if "overlay" in title.lower():
                                logger.debug(f"Ignoring overlay window: {title}")
                                return True

                            if emulator_type != "unknown":
                                try:
                                    # Tentar win32process primeiro; fallback para win32gui
                                    try:
                                        _, pid = win32process.GetWindowThreadProcessId(hwnd)
                                    except AttributeError:
                                        _, pid = win32gui.GetWindowThreadProcessId(hwnd)
                                    process = _psutil.Process(pid)
                                    exe_name = process.name().lower()
                                    logger.debug(f"Window process: {exe_name} (pid: {pid})")
                                    # Verificar se é realmente um emulador pelo exe
                                    if self._is_emulator_process(exe_name, emulator_type):
                                        emulators_list.append(EmulatorInfo(
                                            name=title,
                                            type=emulator_type,
                                            window_title=title,
                                            window_handle=hwnd,
                                            connected=False
                                        ))

                                        logger.info(f"Emulator window detected: {title} (type: {emulator_type}, pid: {pid}, hwnd: {hwnd})")
                                    else:
                                        logger.debug(f"Process {exe_name} does not match emulator type {emulator_type}")
                                except Exception as e:
                                    logger.debug(f"Error analyzing window {hwnd}: {e}", extra={"context": {"hwnd": hwnd, "error_type": type(e).__name__, "error_message": str(e)}})
                except Exception as e:
                    logger.debug(f"Exception in enum_windows_callback for hwnd {hwnd}: {e}", extra={"context": {"hwnd": hwnd, "error_type": type(e).__name__, "error_message": str(e)}})
                    # Ensure the callback never raises
                    return True
                return True

            win32gui.EnumWindows(enum_windows_callback, emulators)
            logger.info(f"Window enumeration completed, found {len(emulators)} emulator windows")

        except Exception as e:
            logger.error(f"Error detecting windows: {e}", extra={"context": {"error_type": type(e).__name__, "error_message": str(e)}})

        # Fallback: detectar por processo se window detection falhar or found nothing
        if not emulators:
            logger.info("No window-detected emulators found; trying process-based detection as fallback")
            logger.debug("Window detection yielded no results, switching to process-based detection")
            emulators = self._detect_by_process()
        
        logger.info(f"Total emulators detected via window detection: {len(emulators)}")
        return emulators
    
    def _identify_emulator_from_title(self, title: str) -> str:
        """Identifica tipo de emulador baseado no título da janela"""
        title_lower = title.lower()
        logger.debug(f"Identifying emulator type from window title: {title[:100]}")
        
        for emulator_type, patterns in self.WINDOW_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, title, re.IGNORECASE):
                    logger.debug(f"Window title matched pattern '{pattern}' for emulator type: {emulator_type}")
                    return emulator_type
        
        logger.debug(f"Window title did not match any known emulator patterns: {title[:100]}")
        return "unknown"
    
    def _is_emulator_process(self, exe_name: str, emulator_type: str) -> bool:
        """Verifica se o processo é realmente um emulador"""
        logger.debug(f"Checking if process '{exe_name}' matches emulator type: {emulator_type}")
        exe_patterns = {
            "bluestacks": ["hd-player.exe", "bluestacks.exe", "bluestacksapp.exe", "bstk"],
            "ldplayer": ["ldplayer.exe", "ldnine.exe"],
            "nox": ["nox.exe", "noxplayer.exe"],
            "memu": ["memu.exe", "memuplayer.exe"]
        }
        
        patterns = exe_patterns.get(emulator_type, [])
        is_emulator = any(pattern in exe_name for pattern in patterns)
        logger.debug(f"Process '{exe_name}' matches emulator type '{emulator_type}': {is_emulator}")
        return is_emulator
    
    def _detect_by_process(self) -> List[EmulatorInfo]:
        """Detecta emuladores por processo (fallback)"""
        logger.info("Starting process-based emulator detection")
        emulators: List[EmulatorInfo] = []
        try:
            import psutil
            logger.info("psutil imported successfully for process detection")
        except ImportError as e:
            logger.warning("psutil não instalado - process detection not available", extra={"context": {"error_type": "ImportError", "error_message": str(e)}})
            logger.info("Install with: pip install psutil")
            return emulators

        # Iterate processes now that psutil is available
        if not hasattr(psutil, 'process_iter'):
            logger.warning("psutil.process_iter não disponível - pulando detecção por processo")
            return emulators

        try:
            logger.debug("Iterating through system processes")
            process_count = 0
            for proc in psutil.process_iter(['pid', 'name', 'exe']):
                try:
                    process_count += 1
                    exe_name = proc.info['name'].lower() if proc.info.get('name') else ""
                    
                    if process_count % 100 == 0:
                        logger.debug(f"Scanned {process_count} processes so far...")

                    # Verificar se é um processo de emulador (exact process name matching only)
                    # NOTE: generic substrings like "ld" cause massive false positives — avoided intentionally
                    emulator_type = None
                    _BLUESTACKS_PROCS = {"hd-player.exe", "bluestacks.exe", "bluestacksapp.exe", "bstkservice.exe"}
                    _LDPLAYER_PROCS = {"ldplayer.exe", "ldnine.exe", "leidian.exe"}
                    _NOX_PROCS = {"nox.exe", "noxplayer.exe", "noxvmsvc.exe"}
                    _MEMU_PROCS = {"memu.exe", "memuplayer.exe", "microvirt.exe"}

                    if exe_name in _BLUESTACKS_PROCS:
                        emulator_type = "bluestacks"
                        logger.debug(f"Found BlueStacks process: {exe_name} (pid: {proc.pid})")
                    elif exe_name in _LDPLAYER_PROCS:
                        emulator_type = "ldplayer"
                        logger.debug(f"Found LDPlayer process: {exe_name} (pid: {proc.pid})")
                    elif exe_name in _NOX_PROCS:
                        emulator_type = "nox"
                        logger.debug(f"Found Nox process: {exe_name} (pid: {proc.pid})")
                    elif exe_name in _MEMU_PROCS:
                        emulator_type = "memu"
                        logger.debug(f"Found MEMU process: {exe_name} (pid: {proc.pid})")

                    if emulator_type:
                        emulators.append(EmulatorInfo(
                            name=proc.info.get('name') or str(proc.pid),
                            type=emulator_type,
                            adb_id=None,
                            window_title=None,
                            window_handle=None,
                            connected=False
                        ))

                        logger.info(f"Emulator process detected: {proc.info.get('name')} (type: {emulator_type}, pid: {proc.pid})")

                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess) as e:
                    logger.debug(f"Skipping process due to {type(e).__name__}: {proc.pid if hasattr(proc, 'pid') else 'unknown'}")
                    continue
            
            logger.info(f"Process scanning completed: scanned {process_count} processes, found {len(emulators)} emulator processes")
        except Exception as e:
            logger.error(f"Error detecting processes: {e}", extra={"context": {"error_type": type(e).__name__, "error_message": str(e)}})

        return emulators
    
    def detect_all(self) -> List[EmulatorInfo]:
        """Detecta todos os emuladores disponíveis (ADB + window)"""
        logger.info("Starting comprehensive emulator detection (ADB + window)")
        self.available_emulators = []
        
        # Detectar via ADB primeiro (preferível)
        logger.info("Step 1: Detecting emulators via ADB")
        adb_emulators = self.detect_adb_devices()
        logger.info(f"ADB detection found {len(adb_emulators)} emulator(s)")
        
        # Detectar via window detection (fallback and for window titles)
        logger.info("Step 2: Detecting emulators via window detection")
        window_emulators = self.detect_window_emulators()
        logger.info(f"Window detection found {len(window_emulators)} emulator(s)")
        
        # Deduplicate emulators based on adb_id or window_title
        logger.info("Step 3: Deduplicating emulators")
        seen_adb_ids = set()
        seen_window_titles = set()
        unique_adb_emulators = []
        
        for ae in adb_emulators:
            if ae.adb_id and ae.adb_id not in seen_adb_ids:
                seen_adb_ids.add(ae.adb_id)
                unique_adb_emulators.append(ae)
                logger.debug(f"Kept unique ADB emulator: {ae.name} (adb_id: {ae.adb_id})")
            elif not ae.adb_id and ae.window_title and ae.window_title not in seen_window_titles:
                seen_window_titles.add(ae.window_title)
                unique_adb_emulators.append(ae)
                logger.debug(f"Kept unique ADB emulator (no adb_id): {ae.name} (window_title: {ae.window_title})")
            else:
                logger.debug(f"Skipped duplicate ADB emulator: {ae.name}")
        
        adb_emulators = unique_adb_emulators
        logger.info(f"After deduplication: {len(adb_emulators)} unique ADB emulator(s)")
        
        # Merge window titles into ADB emulators of the same type
        logger.info("Step 4: Merging ADB and window detection results")
        for ae in adb_emulators:
            # Find matching window emulator by type and title
            matching_window = None
            for we in window_emulators:
                if we.type == ae.type:
                    if ae.adb_id and we.adb_id == ae.adb_id:
                        matching_window = we
                        break
                    elif ae.window_title and we.window_title == ae.window_title:
                        matching_window = we
                        break
                    elif not matching_window:  # Fallback to first match of same type
                        matching_window = we
            
            if matching_window and matching_window.window_title:
                ae.window_title = matching_window.window_title
                ae.window_handle = matching_window.window_handle
                logger.debug(f"Merged window title for {ae.name}: {ae.window_title}")
            else:
                # Provide default window title for common emulators
                if ae.type == "bluestacks":
                    ae.window_title = "BlueStacks App Player"
                    logger.debug(f"Using default window title for BlueStacks: {ae.window_title}")
                elif ae.type == "ldplayer":
                    ae.window_title = "LDPlayer"
                    logger.debug(f"Using default window title for LDPlayer: {ae.window_title}")
            self.available_emulators.append(ae)
            logger.debug(f"Added ADB emulator to final list: {ae.name} (type: {ae.type})")
        
        # Add window-only emulators (not detected via ADB), deduplicated
        logger.info("Step 5: Adding window-only emulators not detected via ADB")
        for we in window_emulators:
            # Skip if already added via ADB
            if we.adb_id and we.adb_id in seen_adb_ids:
                logger.debug(f"Skipped window emulator already in ADB list: {we.name} (adb_id: {we.adb_id})")
                continue
            if we.window_title and we.window_title in seen_window_titles:
                logger.debug(f"Skipped window emulator with duplicate title: {we.name} (window_title: {we.window_title})")
                continue
            if not any(we.window_title == ae.window_title for ae in adb_emulators):
                self.available_emulators.append(we)
                seen_window_titles.add(we.window_title)
                logger.debug(f"Added window-only emulator: {we.name} (type: {we.type})")
        
        logger.info(f"Total emulators detected: {len(self.available_emulators)}")
        logger.info(f"Emulator breakdown: ADB={len(adb_emulators)}, Window={len(window_emulators)}, Final={len(self.available_emulators)}")
        return self.available_emulators
    
    def get_emulator_by_name(self, name: str) -> Optional[EmulatorInfo]:
        """Retorna emulador específico por nome"""
        for emulator in self.available_emulators:
            if emulator.name == name or emulator.window_title == name:
                return emulator
        return None
    
    def get_emulators_by_type(self, emulator_type: str) -> List[EmulatorInfo]:
        """Retorna todos os emuladores de um tipo específico"""
        return [e for e in self.available_emulators if e.type == emulator_type]

    def verify_emulator_5step(self, emulator: EmulatorInfo) -> Dict:
        """
        5-Step Emulator Verification Protocol.

        Step 1: Process alive check  — confirm emulator exe is running
        Step 2: ADB connectivity     — adb connect + adb devices confirms 'device' status
        Step 3: Shell response       — `adb shell echo ping` returns 'ping'
        Step 4: Screen resolution    — `adb shell wm size` returns a parseable resolution
        Step 5: Package check        — `adb shell pm list packages` confirms Android env

        Returns:
            {
              "passed": bool,
              "steps": {1: {...}, 2: {...}, ..., 5: {...}},
              "adb_id": str | None,
              "failure_step": int | None,
            }
        """
        steps: Dict[int, Dict] = {}
        adb_id = emulator.adb_id
        failure_step = None

        adb_path = self._get_bluestacks_adb() if emulator.type == "bluestacks" else get_adb_path()

        # --- Step 1: Process alive ---
        step1 = {"name": "Process alive", "passed": False, "detail": ""}
        try:
            import psutil
            found = False
            target_names = {
                "bluestacks": {"hd-player.exe", "bluestacksapp.exe"},
                "ldplayer": {"ldplayer.exe", "ldnine.exe"},
                "nox": {"nox.exe", "noxplayer.exe"},
                "memu": {"memu.exe", "memuplayer.exe"},
            }.get(emulator.type, set())
            for proc in psutil.process_iter(['name']):
                if proc.info.get('name', '').lower() in target_names:
                    found = True
                    break
            step1["passed"] = found
            step1["detail"] = "Process found" if found else f"No {emulator.type} process running"
        except Exception as e:
            step1["detail"] = f"psutil error: {e}"
        steps[1] = step1
        if not step1["passed"]:
            failure_step = 1
            logger.warning(f"Verification step 1 FAIL for {emulator.name}: {step1['detail']}")
            return {"passed": False, "steps": steps, "adb_id": adb_id, "failure_step": failure_step}

        # --- Step 2: ADB connectivity ---
        step2 = {"name": "ADB connectivity", "passed": False, "detail": ""}
        if adb_id is None:
            step2["detail"] = "No adb_id — run detect_adb_devices() first"
        else:
            try:
                result = subprocess.run(
                    [adb_path, "-s", adb_id, "get-state"],
                    capture_output=True, text=True, timeout=8
                )
                if result.returncode == 0 and "device" in result.stdout.strip():
                    step2["passed"] = True
                    step2["detail"] = result.stdout.strip()
                else:
                    step2["detail"] = f"rc={result.returncode} stdout={result.stdout.strip()[:80]}"
            except Exception as e:
                step2["detail"] = str(e)
        steps[2] = step2
        if not step2["passed"]:
            failure_step = 2
            logger.warning(f"Verification step 2 FAIL for {emulator.name}: {step2['detail']}")
            return {"passed": False, "steps": steps, "adb_id": adb_id, "failure_step": failure_step}

        # --- Step 3: Shell echo ping ---
        step3 = {"name": "Shell response", "passed": False, "detail": ""}
        try:
            result = subprocess.run(
                [adb_path, "-s", adb_id, "shell", "echo", "ping"],
                capture_output=True, text=True, timeout=8
            )
            out = result.stdout.strip()
            step3["passed"] = "ping" in out.lower()
            step3["detail"] = out[:80] if out else f"rc={result.returncode}"
        except Exception as e:
            step3["detail"] = str(e)
        steps[3] = step3
        if not step3["passed"]:
            failure_step = 3
            logger.warning(f"Verification step 3 FAIL for {emulator.name}: {step3['detail']}")
            return {"passed": False, "steps": steps, "adb_id": adb_id, "failure_step": failure_step}

        # --- Step 4: Screen resolution ---
        step4 = {"name": "Screen resolution", "passed": False, "detail": ""}
        try:
            result = subprocess.run(
                [adb_path, "-s", adb_id, "shell", "wm", "size"],
                capture_output=True, text=True, timeout=8
            )
            out = result.stdout.strip()
            import re as _re
            if _re.search(r'\d+x\d+', out):
                step4["passed"] = True
                step4["detail"] = out[:80]
            else:
                step4["detail"] = f"Unexpected output: {out[:80]}"
        except Exception as e:
            step4["detail"] = str(e)
        steps[4] = step4
        if not step4["passed"]:
            failure_step = 4
            logger.warning(f"Verification step 4 FAIL for {emulator.name}: {step4['detail']}")
            return {"passed": False, "steps": steps, "adb_id": adb_id, "failure_step": failure_step}

        # --- Step 5: Android package list ---
        step5 = {"name": "Package list (Android env)", "passed": False, "detail": ""}
        try:
            result = subprocess.run(
                [adb_path, "-s", adb_id, "shell", "pm", "list", "packages", "-3"],
                capture_output=True, text=True, timeout=12
            )
            out = result.stdout.strip()
            # Any output means Android PM is responding
            step5["passed"] = result.returncode == 0 and len(out) > 0
            step5["detail"] = f"{len(out.splitlines())} packages listed" if step5["passed"] else f"rc={result.returncode}"
        except Exception as e:
            step5["detail"] = str(e)
        steps[5] = step5
        if not step5["passed"]:
            failure_step = 5
            logger.warning(f"Verification step 5 FAIL for {emulator.name}: {step5['detail']}")
            return {"passed": False, "steps": steps, "adb_id": adb_id, "failure_step": failure_step}

        logger.info(f"5-step verification PASSED for {emulator.name} (adb_id={adb_id})")
        return {"passed": True, "steps": steps, "adb_id": adb_id, "failure_step": None}


# Singleton instance
_detector_instance: Optional[EmulatorDetector] = None


def get_emulator_detector() -> EmulatorDetector:
    """Retorna instância singleton do detetor"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = EmulatorDetector()
    return _detector_instance


def get_detection_report() -> Dict:
    """Gera um relatório de detecção contendo lista de emuladores e campos de telemetria.

    Retorna um dicionário com as chaves:
      - emulators: lista de emuladores (dicts)
      - pywin32_available: bool
      - psutil_available: bool
      - chosen_adb_path: str
    """
    detector = get_emulator_detector()

    # Ensure we have a deterministic chosen adb path
    chosen_adb_path = get_adb_path()

    # Check optional dependencies availability without raising
    pywin32_available = False
    try:
        import win32gui  # type: ignore
        pywin32_available = True
    except Exception:
        pywin32_available = False

    psutil_available = False
    try:
        import psutil  # type: ignore
        psutil_available = True
    except Exception:
        psutil_available = False

    # Perform detection (may be empty if tools are absent)
    emulators = []
    try:
        emulators = detector.detect_all()
    except Exception as e:
        logger.debug(f"Non-fatal error while running detect_all: {e}")

    # Convert dataclass objects to plain dicts for deterministic JSON-like output
    emu_list = []
    for e in emulators:
        emu_list.append({
            "name": e.name,
            "type": e.type,
            "adb_id": e.adb_id,
            "window_title": e.window_title,
            "window_handle": e.window_handle,
            "connected": e.connected,
        })

    report = {
        "emulators": emu_list,
        "pywin32_available": bool(pywin32_available),
        "psutil_available": bool(psutil_available),
        "chosen_adb_path": str(chosen_adb_path),
    }

    # If installation report is available nearby, include a deterministic pointer for diagnostics
    try:
        installation_report = Path(__file__).parent / 'installation_report.json'
        if installation_report.exists():
            report['installation_report_path'] = str(installation_report)
    except Exception:
        pass

    return report
