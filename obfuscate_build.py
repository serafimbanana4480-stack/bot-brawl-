"""
obfuscate_build.py

Seguranca: Ofuscacao de Codigo para Soberana Omega.

Codigo Python aberto e risco (easy reverse engineering). Este modulo
prepara o build para distribuicao segura:

- Usa PyArmor para ofuscacao avancada
- Fallback para compilacao com Cython (mais rapido e seguro)
- Remove docstrings e comentarios de producao
- Verifica integridade do build

ATENCAO: Nao ofusca segredos — use variaveis de ambiente para isso.
"""

import os
import sys
import shutil
import logging
import subprocess
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)


class BuildObfuscator:
    """
    Prepara build ofuscado do bot.

    Uso:
        obf = BuildObfuscator()
        obf.build_secure("./dist/soberana_omega")
    """

    def __init__(self, source_dir: Path = Path("."), output_dir: Path = Path("dist/secure")):
        self.source_dir = Path(source_dir).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.exclude_patterns = [
            "*.pyc", "__pycache__", "*.pyo", ".git", ".gitignore",
            ".venv", "venv", "env", "*.md", "tests", "test_*",
            "logs", "data", "tmp", "tmp2", "tmp3",
            ".planning", ".codeium", ".claude", ".cursor",
        ]

    def build_with_pyarmor(self, entry_script: str = "wrapper.py") -> bool:
        """Usa PyArmor para ofuscar codigo Python."""
        try:
            import pyarmor  # noqa: F401
        except ImportError:
            logger.warning("[OBFUSCATE] PyArmor nao instalado. Use: pip install pyarmor")
            return False

        logger.info("[OBFUSCATE] Iniciando build com PyArmor")

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, "-m", "pyarmor", "obfuscate",
            "--restrict", "--advanced", "2",
            "--output", str(self.output_dir),
            entry_script,
        ]

        try:
            result = subprocess.run(cmd, cwd=self.source_dir, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("[OBFUSCATE] PyArmor completo: %s", self.output_dir)
                return True
            else:
                logger.error("[OBFUSCATE] PyArmor falhou: %s", result.stderr)
                return False
        except Exception as e:
            logger.error("[OBFUSCATE] Erro ao executar PyArmor: %s", e)
            return False

    def build_with_cython(self, target_modules: Optional[List[str]] = None) -> bool:
        """Compila modulos Python com Cython para .pyd/.so."""
        try:
            import Cython  # noqa: F401
        except ImportError:
            logger.warning("[OBFUSCATE] Cython nao instalado. Use: pip install cython")
            return False

        logger.info("[OBFUSCATE] Iniciando build com Cython")
        modules = target_modules or [
            "wrapper.py", "pylaai_real/play.py", "pylaai_real/movement.py",
            "decision/utility_ai.py", "core/error_recovery.py", "vision/detect.py",
        ]

        setup_script = self._generate_cython_setup(modules)
        setup_path = self.output_dir / "setup_cython.py"
        with open(setup_path, "w", encoding="utf-8") as f:
            f.write(setup_script)

        try:
            result = subprocess.run(
                [sys.executable, str(setup_path), "build_ext", "--inplace"],
                cwd=self.source_dir, capture_output=True, text=True, timeout=300,
            )
            if result.returncode == 0:
                logger.info("[OBFUSCATE] Cython completo")
                return True
            else:
                logger.error("[OBFUSCATE] Cython falhou: %s", result.stderr)
                return False
        except Exception as e:
            logger.error("[OBFUSCATE] Erro ao executar Cython: %s", e)
            return False

    def _generate_cython_setup(self, modules: List[str]) -> str:
        module_list = ",\n        ".join(f'"{m.replace("/", ".").replace("\\", ".").replace(".py", "")}"' for m in modules)
        return f'''\
from setuptools import setup
from Cython.Build import cythonize
from Cython.Compiler import Options

Options.docstrings = False

setup(
    ext_modules=cythonize(
        [{module_list}],
        compiler_directives={{
            "language_level": "3",
            "annotation_typing": False,
        }},
        exclude=["**/tests/**", "**/test_*"],
    ),
    zip_safe=False,
)
'''

    def build_secure(self, method: str = "auto") -> Dict[str, any]:
        """Build completo de seguranca."""
        results = {"method_used": None, "success": False, "output_dir": str(self.output_dir), "warnings": []}

        if method in ("pyarmor", "auto"):
            if self.build_with_pyarmor():
                results["method_used"] = "pyarmor"
                results["success"] = True
            elif method == "auto":
                results["warnings"].append("PyArmor falhou, tentando Cython")
                if self.build_with_cython():
                    results["method_used"] = "cython"
                    results["success"] = True
                else:
                    results["warnings"].append("Cython tambem falhou")
            else:
                results["warnings"].append("PyArmor falhou")
        elif method == "cython":
            if self.build_with_cython():
                results["method_used"] = "cython"
                results["success"] = True
            else:
                results["warnings"].append("Cython falhou")

        logger.info("[OBFUSCATE] Build finalizado: %s", results)
        return results

    def verify_build(self) -> bool:
        """Verifica se build esta funcional."""
        if not self.output_dir.exists():
            return False
        entry_points = (
            list(self.output_dir.glob("**/wrapper.py")) +
            list(self.output_dir.glob("**/wrapper.pyd")) +
            list(self.output_dir.glob("**/wrapper.so"))
        )
        if not entry_points:
            logger.error("[OBFUSCATE] Entry point nao encontrado no build")
            return False
        logger.info("[OBFUSCATE] Build verificado: %d entry points", len(entry_points))
        return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    obf = BuildObfuscator()
    result = obf.build_secure(method="auto")
    print("Resultado:", result)
