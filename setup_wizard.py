"""
setup_wizard.py

Wizard interativo de configuração inicial do bot.
Guia o usuário através da configuração passo-a-passo.

Funcionalidades:
- Detecção automática de emulador
- Teste de conexão ADB
- Calibração interativa de coordenadas
- Geração automática de config.json
- Teste de componentes
- Validação de configuração
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import subprocess

# Adicionar projeto ao path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)


class SetupWizard:
    """Wizard de configuração inicial."""
    
    def __init__(self):
        self.config = {}
        self.calibrated_coords = {}
        self.project_root = project_root
        
        # Configurar logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def run(self):
        """Executa o wizard completo."""
        
        print("=" * 70)
        print(" SOBERANA OMEGA BRAWL STARS BOT - WIZARD DE CONFIGURAÇÃO")
        print("=" * 70)
        print()
        
        try:
            # Passo 1: Verificar dependências
            self._check_dependencies()
            
            # Passo 2: Detectar emulador
            self._detect_emulator()
            
            # Passo 3: Configurar ADB
            self._configure_adb()
            
            # Passo 4: Calibrar coordenadas
            self._calibrate_coordinates()
            
            # Passo 5: Configurar parâmetros de jogo
            self._configure_game_parameters()
            
            # Passo 6: Configurar segurança
            self._configure_safety()
            
            # Passo 7: Salvar configuração
            self._save_config()
            
            # Passo 8: Testar configuração
            self._test_config()
            
            print()
            print("=" * 70)
            print(" CONFIGURAÇÃO CONCLUÍDA COM SUCESSO!")
            print("=" * 70)
            print()
            print("Próximos passos:")
            print("1. Execute: python bot.py")
            print("2. Abra o dashboard em: http://localhost:8765")
            print("3. Monitore os primeiros minutos de operação")
            print()
            print("Para problemas, consulte: INSTALLATION_GUIDE.md")
            print()
        
        except KeyboardInterrupt:
            print("\n\nSetup cancelado pelo usuário.")
            sys.exit(1)
        except Exception as e:
            print(f"\n\nErro durante setup: {e}")
            logger.error(f"Setup error: {e}", exc_info=True)
            sys.exit(1)
    
    def _check_dependencies(self):
        """Verifica se todas as dependências estão instaladas."""
        
        print("📦 Verificando dependências...")
        
        required_packages = [
            "cv2",  # opencv
            "numpy",
            "torch",
            "ultralytics"
        ]
        
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
                print(f"  ✓ {package}")
            except ImportError:
                print(f"  ✗ {package} (não encontrado)")
                missing.append(package)
        
        # Verificar EasyOCR (opcional)
        try:
            import easyocr
            print(f"  ✓ easyocr (opcional)")
        except ImportError:
            print(f"  ○ easyocr (opcional, não instalado)")
        
        if missing:
            print(f"\n⚠️  Pacotes faltando: {', '.join(missing)}")
            print("Instale com: pip install -r requirements.txt")
            response = input("Continuar mesmo assim? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        print()
    
    def _detect_emulator(self):
        """Detecta emulador instalado."""
        
        print("🎮 Detectando emulador...")
        
        # Tentar detectar LDPlayer
        ldplayer_detected = self._detect_ldplayer()
        
        # Tentar detectar BlueStacks
        bluestacks_detected = self._detect_bluestacks()
        
        if ldplayer_detected:
            self.config["emulator"] = {
                "type": "ldplayer",
                "adb_port": 5555,
                "window_title": "LDPlayer"
            }
            print("  ✓ LDPlayer detectado")
        elif bluestacks_detected:
            self.config["emulator"] = {
                "type": "bluestacks",
                "adb_port": 5555,
                "window_title": "BlueStacks App Player"
            }
            print("  ✓ BlueStacks detectado")
        else:
            print("  ⚠️  Nenhum emulador detectado automaticamente")
            print()
            print("Emuladores suportados:")
            print("  1. LDPlayer")
            print("  2. BlueStacks")
            
            choice = input("Selecione o emulador (1/2): ")
            
            if choice == "1":
                self.config["emulator"] = {
                    "type": "ldplayer",
                    "adb_port": 5555,
                    "window_title": "LDPlayer"
                }
            elif choice == "2":
                self.config["emulator"] = {
                    "type": "bluestacks",
                    "adb_port": 5555,
                    "window_title": "BlueStacks App Player"
                }
            else:
                print("Emulador inválido. Usando LDPlayer como default.")
                self.config["emulator"] = {
                    "type": "ldplayer",
                    "adb_port": 5555,
                    "window_title": "LDPlayer"
                }
        
        # Perguntar sobre resolução
        print()
        print("Resolução do emulador:")
        print("  1. 1920x1080 (recomendado)")
        print("  2. 1280x720")
        print("  3. Outro")
        
        res_choice = input("Selecione resolução (1/2/3): ")
        
        if res_choice == "1":
            self.config["emulator"]["resolution"] = [1920, 1080]
            self.config["emulator"]["dpi"] = 280
        elif res_choice == "2":
            self.config["emulator"]["resolution"] = [1280, 720]
            self.config["emulator"]["dpi"] = 240
        else:
            width = int(input("Largura: "))
            height = int(input("Altura: "))
            self.config["emulator"]["resolution"] = [width, height]
            self.config["emulator"]["dpi"] = 280
        
        print()
    
    def _detect_ldplayer(self) -> bool:
        """Tenta detectar LDPlayer."""
        try:
            # Verificar processo do LDPlayer
            result = subprocess.run(
                ["tasklist"], 
                capture_output=True, 
                text=True
            )
            return "LDPlayer" in result.stdout
        except:
            return False
    
    def _detect_bluestacks(self) -> bool:
        """Tenta detectar BlueStacks."""
        try:
            result = subprocess.run(
                ["tasklist"], 
                capture_output=True, 
                text=True
            )
            return "BlueStacks" in result.stdout
        except:
            return False
    
    def _configure_adb(self):
        """Configura e testa conexão ADB."""
        
        print("🔌 Configurando ADB...")
        
        # Verificar se ADB está instalado
        try:
            result = subprocess.run(
                ["adb", "version"], 
                capture_output=True, 
                text=True
            )
            print(f"  ✓ ADB encontrado")
        except FileNotFoundError:
            print("  ✗ ADB não encontrado")
            print()
            print("Por favor, instale ADB:")
            print("  1. Download: https://developer.android.com/studio/releases/platform-tools")
            print("  2. Extraia para C:\\platform-tools")
            print("  3. Adicione C:\\platform-tools ao PATH do sistema")
            print()
            input("Pressione Enter quando ADB estiver instalado...")
        
        # Testar conexão
        print("  Testando conexão com emulador...")
        
        adb_port = self.config["emulator"]["adb_port"]
        device_id = f"emulator-{adb_port}"
        
        try:
            # Tentar conectar
            subprocess.run(
                ["adb", "connect", device_id],
                capture_output=True
            )
            
            # Listar dispositivos
            result = subprocess.run(
                ["adb", "devices"],
                capture_output=True,
                text=True
            )
            
            if device_id in result.stdout:
                print(f"  ✓ Conectado ao emulador ({device_id})")
            else:
                print(f"  ⚠️  Não foi possível conectar ao emulador")
                print(f"  Verifique se o emulador está rodando e o ADB está habilitado")
                response = input("Continuar mesmo assim? (y/n): ")
                if response.lower() != 'y':
                    sys.exit(1)
        
        except Exception as e:
            print(f"  ✗ Erro ao testar ADB: {e}")
            response = input("Continuar mesmo assim? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
        
        print()
    
    def _calibrate_coordinates(self):
        """Calibra coordenadas de forma interativa."""
        
        print("🎯 Calibrando coordenadas...")
        print()
        print("Este passo requer que você clique nos elementos na tela do emulador.")
        print("Certifique-se de que o Brawl Stars está aberto no lobby principal.")
        print()
        
        try:
            from pylaai_real.screenshot_taker import ScreenshotTaker
            from pylaai_real.auto_calibrator import interactive_calibration_setup
        except ImportError as e:
            print(f"  ✗ Não foi possível importar módulos de calibração: {e}")
            print("  Pulando calibração interativa...")
            print()
            return
        
        # Criar screenshot taker
        try:
            screenshot = ScreenshotTaker()
            print("  ✓ Captura de tela funcionando")
        except Exception as e:
            print(f"  ✗ Erro ao inicializar captura de tela: {e}")
            print("  Usando coordenadas padrão...")
            self._use_default_coordinates()
            return
        
        # Elementos para calibrar
        elements = [
            "play_button",
            "brawl_stars_logo",
            "x_button"
        ]
        
        # Executar calibração
        try:
            coords = interactive_calibration_setup(screenshot, elements)
            self.calibrated_coords = coords
            
            print()
            print("  Coordenadas calibradas:")
            for element, (x, y) in coords.items():
                print(f"    {element}: ({x}, {y})")
            
        except Exception as e:
            print(f"  ⚠️  Calibração interativa falhou: {e}")
            print("  Usando coordenadas padrão...")
            self._use_default_coordinates()
        
        print()
    
    def _use_default_coordinates(self):
        """Usa coordenadas padrão baseadas na resolução."""
        
        resolution = self.config["emulator"]["resolution"]
        w, h = resolution
        
        # Coordenadas normalizadas (0-1)
        normalized_coords = {
            "play_button": (0.94, 0.89),
            "brawl_stars_logo": (0.5, 0.1),
            "x_button": (0.97, 0.06)
        }
        
        # Converter para coordenadas absolutas
        for element, (nx, ny) in normalized_coords.items():
            self.calibrated_coords[element] = (int(nx * w), int(ny * h))
        
        print("  Coordenadas padrão aplicadas")
    
    def _configure_game_parameters(self):
        """Configura parâmetros do jogo."""
        
        print("🎮 Configurando parâmetros do jogo...")
        
        # Modo de jogo
        print()
        print("Modo de jogo:")
        print("  1. Gem Grab (recomendado)")
        print("  2. Showdown")
        print("  3. Brawl Ball")
        print("  4. Heist")
        
        mode_choice = input("Selecione modo (1/2/3/4): ")
        
        modes = {
            "1": "gem_grab",
            "2": "showdown",
            "3": "brawl_ball",
            "4": "heist"
        }
        
        self.config["game"] = {
            "mode": modes.get(mode_choice, "gem_grab"),
            "brawler": "colt",  # Default
            "language": "en",
            "resolution": f"{self.config['emulator']['resolution'][0]}x{self.config['emulator']['resolution'][1]}"
        }
        
        # Brawler
        print()
        print("Brawler inicial (recomendados: colt, shelly, el_primo):")
        brawler = input("Brawler (default: colt): ") or "colt"
        self.config["game"]["brawler"] = brawler.lower()
        
        print()
    
    def _configure_safety(self):
        """Configura parâmetros de segurança."""
        
        print("🛡️  Configurando segurança...")
        
        # Troféus
        print()
        max_trophies = int(input("Limite de troféus (default: 400): ") or "400")
        warning_trophies = int(input("Aviso de troféus (default: 380): ") or "380")
        
        # Sessão
        max_hours = float(input("Máx horas por sessão (default: 3.0): ") or "3.0")
        
        # APM
        min_apm = int(input("APM mínimo (default: 20): ") or "20")
        max_apm = int(input("APM máximo (default: 60): ") or "60")
        
        self.config["safety"] = {
            "max_trophies": max_trophies,
            "warning_trophies": warning_trophies,
            "max_session_hours": max_hours,
            "min_apm": min_apm,
            "max_apm": max_apm,
            "break_duration_min": 300,
            "break_duration_max": 900,
            "auto_stop_on_detection": True,
            "suspicious_pattern_threshold": 5
        }
        
        # Humanização
        self.config["humanization"] = {
            "enabled": True,
            "mouse_curve": True,
            "random_delays": True,
            "min_delay_ms": 50,
            "max_delay_ms": 200,
            "bezier_control_points": 3,
            "jitter_range": 2
        }
        
        print()
    
    def _save_config(self):
        """Salva configuração em arquivo."""
        
        print("💾 Salvando configuração...")
        
        # Configuração completa
        full_config = {
            "version": "1.0.0",
            **self.config,
            "vision": {
                "main_model": "brawlstars_yolov8.pt",
                "brawler_id_model": "brawler_id.pt",
                "confidence_threshold": 0.37,
                "nms_iou_threshold": 0.45,
                "classes": ["Player", "Bush", "Enemy", "Cubebox"],
                "thresholds": [0.37, 0.47, 0.57, 0.65]
            },
            "api": {
                "host": "127.0.0.1",
                "port": 8003,
                "cors_origins": ["http://localhost:3000"],
                "enable_swagger": True
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "file": "logs/brawl_bot.log",
                "max_file_size_mb": 10,
                "backup_count": 5
            },
            "performance": {
                "target_fps": 30,
                "screenshot_interval_ms": 100,
                "action_cooldown_ms": 50
            },
            "combat": {
                "enable_prediction": True,
                "prediction_time_ahead": 0.2,
                "enable_aim_assist": True,
                "aim_smoothing": True,
                "enable_abilities": True,
                "gadget_cooldown": 15,
                "enable_power_cube_collection": True,
                "power_cube_safe_distance": 300,
                "tactical_movement": True,
                "aggression_level": 0.7
            },
            "dashboard": {
                "enabled": True,
                "port": 8765,
                "auto_start": True,
                "replay_auto_record": False,
                "replay_max_frames": 150
            }
        }
        
        # Criar diretórios necessários
        self.project_root.joinpath("logs").mkdir(exist_ok=True)
        self.project_root.joinpath("data").mkdir(exist_ok=True)
        self.project_root.joinpath("models").mkdir(exist_ok=True)
        
        # Salvar config.json
        config_path = self.project_root / "config.json"
        with open(config_path, 'w') as f:
            json.dump(full_config, f, indent=2)
        
        print(f"  ✓ Configuração salva em: {config_path}")
        
        # Salvar coordenadas calibradas
        if self.calibrated_coords:
            coords_path = self.project_root / "data" / "calibrated_coords.json"
            with open(coords_path, 'w') as f:
                json.dump(self.calibrated_coords, f, indent=2)
            print(f"  ✓ Coordenadas salvas em: {coords_path}")
        
        print()
    
    def _test_config(self):
        """Testa a configuração."""
        
        print("🧪 Testando configuração...")
        
        try:
            # Testar import do wrapper
            from brawl_bot.wrapper import PylaAIEnhanced
            print("  ✓ Wrapper importado com sucesso")
            
            # Testar criação do bot (sem iniciar)
            bot = PylaAIEnhanced()
            print("  ✓ Bot criado com sucesso")
            
            # Verificar componentes
            if bot.screenshot:
                print("  ✓ ScreenshotTaker inicializado")
            if bot.state_manager:
                print("  ✓ StateManager inicializado")
            if bot.play_logic:
                print("  ✓ PlayLogic inicializado")
            
            print()
            print("✓ Todos os testes passaram!")
            
        except Exception as e:
            print(f"  ✗ Erro no teste: {e}")
            print()
            print("⚠️  Alguns componentes falharam, mas o bot pode funcionar parcialmente.")
            print("Consulte INSTALLATION_GUIDE.md para troubleshooting.")
        
        print()


def main():
    """Função principal."""
    wizard = SetupWizard()
    wizard.run()


if __name__ == "__main__":
    main()
