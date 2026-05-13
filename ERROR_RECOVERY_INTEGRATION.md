# Integração do Sistema de Error Recovery ao Wrapper

## 📋 Visão Geral

O sistema de error recovery (`core/error_recovery.py`) fornece tratamento robusto de erros com recovery automático. Este guia mostra como integrá-lo ao `wrapper.py`.

---

## 🔧 Integração Básica

### 1. Importar e Inicializar no Wrapper

```python
# Em wrapper.py

from core.error_recovery import (
    ErrorRecoverySystem,
    ErrorRecoveryIntegration,
    with_error_recovery,
    ErrorType,
    ErrorSeverity
)

class PylaAIEnhanced:
    def __init__(self, ...):
        # ... código existente ...
        
        # Inicializar sistema de error recovery
        self.error_recovery = ErrorRecoverySystem(
            enable_auto_recovery=True,
            max_recovery_attempts=3,
            global_circuit_breaker=True
        )
        
        # Criar integração
        self.recovery_integration = ErrorRecoveryIntegration(self)
        
        # Habilitar automaticamente
        self.recovery_integration.enable()
        
        logger.info("[WRAPPER] Error recovery system inicializado")
```

### 2. Envolver Métodos Críticos

```python
def __init__(self, ...):
    # ... código existente ...
    
    # Após inicializar componentes
    self._setup_error_recovery()

def _setup_error_recovery(self):
    """Configura tratamento de erro para métodos críticos."""
    
    # Envolver métodos específicos
    self.recovery_integration.wrap_method(
        "capture_screenshot", 
        component="screenshot", 
        operation="capture"
    )
    
    self.recovery_integration.wrap_method(
        "detect_state",
        component="state_manager",
        operation="detect"
    )
    
    self.recovery_integration.wrap_method(
        "execute_action",
        component="emulator_controller",
        operation="execute"
    )
    
    # Envolver loop principal
    self.recovery_integration.wrap_main_loop()
```

### 3. Modificar Loop Principal com Try-Catch Granular

```python
def _main_loop(self):
    """Loop principal com tratamento de erro granular."""
    
    while self.running:
        try:
            # 1. Capturar screenshot
            screenshot = self._safe_capture_screenshot()
            
            # 2. Detectar estado
            state, confidence = self._safe_detect_state(screenshot)
            
            # 3. Atualizar sistema de recuperação de estado
            if hasattr(self, 'state_recovery'):
                self.state_recovery.update_state(state, confidence)
            
            # 4. Executar handler do estado
            self._safe_execute_state_handler(state, screenshot)
            
            # 5. Atualizar métricas
            self._safe_update_metrics()
            
            # 6. Aplicar delay adaptativo
            delay = self._get_adaptive_delay(state)
            time.sleep(delay)
        
        except MemoryError as e:
            # Tratar erro de memória especificamente
            context = self.error_recovery.classify_error(
                e, 
                component="wrapper", 
                operation="main_loop"
            )
            self.error_recovery.handle_error(context, self)
            
            # Limpar memória
            import gc
            gc.collect()
            
            # Continuar loop
            continue
        
        except KeyboardInterrupt:
            # Permitir interrupção limpa
            logger.info("[WRAPPER] Interrupção pelo usuário")
            break
        
        except Exception as e:
            # Erro genérico - tentar recovery
            context = self.error_recovery.classify_error(
                e,
                component="wrapper",
                operation="main_loop"
            )
            
            recovered = self.error_recovery.handle_error(context, self)
            
            if not recovered:
                # Se não recuperou, esperar e tentar novamente
                logger.error(f"[WRAPPER] Erro não recuperado: {e}")
                time.sleep(5.0)
```

### 4. Implementar Métodos "Safe" para Operações Críticas

```python
def _safe_capture_screenshot(self) -> np.ndarray:
    """Captura screenshot com tratamento de erro."""
    
    max_attempts = 3
    
    for attempt in range(max_attempts):
        try:
            if self.screenshot:
                return self.screenshot.capture()
            else:
                raise RuntimeError("ScreenshotTaker não inicializado")
        
        except Exception as e:
            logger.warning(f"[WRAPPER] Falha na captura (tentativa {attempt + 1}/{max_attempts}): {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(0.5)
            else:
                # Última tentativa falhou - tentar recovery
                context = self.error_recovery.classify_error(
                    e,
                    component="screenshot",
                    operation="capture"
                )
                
                recovered = self.error_recovery.handle_error(context, self)
                
                if not recovered:
                    raise RuntimeError("Falha crítica na captura de screenshot")

def _safe_detect_state(self, screenshot: np.ndarray) -> Tuple[str, float]:
    """Detecta estado com tratamento de erro."""
    
    max_attempts = 2
    
    for attempt in range(max_attempts):
        try:
            if self.unified_detector:
                result = self.unified_detector.detect(screenshot)
                return result.state, result.confidence
            elif self.state_finder:
                result = self.state_finder.find_state(screenshot)
                return result, 0.7  # Confiança padrão
            else:
                return "unknown", 0.0
        
        except Exception as e:
            logger.warning(f"[WRAPPER] Falha na detecção (tentativa {attempt + 1}/{max_attempts}): {e}")
            
            if attempt < max_attempts - 1:
                time.sleep(0.3)
            else:
                # Usar fallback
                context = self.error_recovery.classify_error(
                    e,
                    component="state_detector",
                    operation="detect"
                )
                
                recovered = self.error_recovery.handle_error(context, self)
                
                if not recovered:
                    return "unknown", 0.0

def _safe_execute_state_handler(self, state: str, screenshot: np.ndarray):
    """Executa handler de estado com tratamento de erro."""
    
    try:
        handler = self.state_manager.states.get(state)
        if handler:
            if state in ["in_game", "unknown"]:
                # Estes handlers recebem screenshot
                handler(screenshot)
            else:
                handler()
    
    except Exception as e:
        logger.error(f"[WRAPPER] Erro ao executar handler para estado {state}: {e}")
        
        context = self.error_recovery.classify_error(
            e,
            component="state_manager",
            operation=f"handle_{state}"
        )
        
        self.error_recovery.handle_error(context, self)

def _safe_update_metrics(self):
    """Atualiza métricas com tratamento de erro."""
    
    try:
        if self.observability:
            # Atualizar métricas normalmente
            pass
    
    except Exception as e:
        logger.debug(f"[WRAPPER] Erro ao atualizar métricas (não crítico): {e}")
        # Não propagar erro - métricas não são críticas
```

---

## 🎯 Exemplo Completo de Integração

Aqui está um exemplo completo de como modificar o `wrapper.py`:

```python
"""
wrapper.py - COM ERROR RECOVERY INTEGRADO

Wrapper que integra PylaAI real com Safety System, Humanização e Error Recovery.
"""

import time
import json
import threading
import os
import signal
import base64
from pathlib import Path
from typing import Optional, Dict, List, Any
import logging

# Imports existentes
from .pylaai_real.state_finder import StateFinder
from .pylaai_real.state_manager import StateManager
from .pylaai_real.screenshot_taker import ScreenshotTaker
# ... outros imports ...

# NOVO: Import de error recovery
from core.error_recovery import (
    ErrorRecoverySystem,
    ErrorRecoveryIntegration,
    with_error_recovery
)

logger = logging.getLogger(__name__)


class PylaAIEnhanced:
    """
    PylaAI com melhorias:
    - Safety System
    - Humanização
    - Error Recovery System (NOVO)
    - Fila de brawlers
    - Integração EmulatorController
    """
    
    def __init__(
        self,
        install_path: Path = _DEFAULT_INSTALL_PATH,
        safety_config: Optional[SafetyConfig] = None,
        humanization_config: Optional[HumanizationConfig] = None,
        enable_error_recovery: bool = True  # NOVO parâmetro
    ):
        logger.info("[WRAPPER] Inicializando PylaAIEnhanced")
        
        # ... código existente de inicialização ...
        
        # NOVO: Inicializar sistema de error recovery
        self.enable_error_recovery = enable_error_recovery
        if enable_error_recovery:
            self.error_recovery = ErrorRecoverySystem(
                enable_auto_recovery=True,
                max_recovery_attempts=3,
                global_circuit_breaker=True
            )
            
            self.recovery_integration = ErrorRecoveryIntegration(self)
            self.recovery_integration.enable()
            
            logger.info("[WRAPPER] Error recovery system habilitado")
        
        # ... resto da inicialização ...
        
        # Configurar error recovery após componentes inicializados
        if enable_error_recovery:
            self._setup_error_recovery()
    
    def _setup_error_recovery(self):
        """Configura tratamento de erro para métodos críticos."""
        
        if not hasattr(self, 'recovery_integration'):
            return
        
        # Envolver métodos críticos
        self.recovery_integration.wrap_method(
            "capture_screenshot",
            component="screenshot",
            operation="capture"
        )
        
        self.recovery_integration.wrap_method(
            "detect_state",
            component="state_manager",
            operation="detect"
        )
        
        # Envolver loop principal
        self.recovery_integration.wrap_main_loop()
        
        logger.info("[WRAPPER] Error recovery configurado para métodos críticos")
    
    def start(self):
        """Inicia o bot com error recovery."""
        
        logger.info("[WRAPPER] Iniciando bot...")
        
        # ... código existente de start ...
        
        # NOVO: Iniciar thread de monitoramento de erros
        if self.enable_error_recovery:
            self._error_monitor_thread = threading.Thread(
                target=self._error_monitor_loop,
                daemon=True
            )
            self._error_monitor_thread.start()
            logger.info("[WRAPPER] Error monitor thread iniciado")
        
        # ... restante do start ...
    
    def _main_loop(self):
        """Loop principal com tratamento de erro granular."""
        
        logger.info("[WRAPPER] Iniciando loop principal")
        
        while self.running:
            cycle_start = time.time()
            
            try:
                # 1. Capturar screenshot (com tratamento de erro)
                screenshot = self._safe_capture_screenshot()
                
                # 2. Detectar estado (com tratamento de erro)
                state, confidence = self._safe_detect_state(screenshot)
                
                # 3. Atualizar sistema de recuperação de estado
                if hasattr(self, 'state_recovery'):
                    self.state_recovery.update_state(state, confidence)
                    
                    # Se está em recuperação, executar e continuar
                    if self.state_recovery.is_recovering():
                        self.state_recovery.execute_recovery_step()
                        continue
                
                # 4. Executar handler do estado (com tratamento de erro)
                self._safe_execute_state_handler(state, screenshot)
                
                # 5. Atualizar métricas
                self._safe_update_metrics()
                
                # 6. Calcular e aplicar delay adaptativo
                cycle_time = time.time() - cycle_start
                delay = self._get_adaptive_delay(state, cycle_time)
                time.sleep(delay)
            
            except MemoryError as e:
                # Tratar erro de memória especificamente
                logger.error(f"[WRAPPER] Erro de memória: {e}")
                
                if self.enable_error_recovery:
                    context = self.error_recovery.classify_error(
                        e, component="wrapper", operation="main_loop"
                    )
                    self.error_recovery.handle_error(context, self)
                
                # Limpar memória
                import gc
                gc.collect()
                time.sleep(2.0)
                continue
            
            except KeyboardInterrupt:
                logger.info("[WRAPPER] Interrupção pelo usuário")
                break
            
            except Exception as e:
                # Erro genérico
                logger.error(f"[WRAPPER] Erro no loop principal: {e}")
                
                if self.enable_error_recovery:
                    context = self.error_recovery.classify_error(
                        e, component="wrapper", operation="main_loop"
                    )
                    recovered = self.error_recovery.handle_error(context, self)
                    
                    if recovered:
                        logger.info("[WRAPPER] Recovery bem-sucedido, continuando loop")
                        continue
                
                # Se não recuperou, esperar e tentar novamente
                logger.warning("[WRAPPER] Aguardando 5s antes de tentar novamente...")
                time.sleep(5.0)
    
    def _safe_capture_screenshot(self) -> np.ndarray:
        """Captura screenshot com retry e recovery."""
        
        max_attempts = 3
        
        for attempt in range(max_attempts):
            try:
                if self.screenshot:
                    return self.screenshot.capture()
                else:
                    raise RuntimeError("ScreenshotTaker não inicializado")
            
            except Exception as e:
                logger.warning(f"[WRAPPER] Falha na captura ({attempt + 1}/{max_attempts}): {e}")
                
                if attempt < max_attempts - 1:
                    time.sleep(0.5)
                else:
                    # Tentar recovery
                    if self.enable_error_recovery:
                        context = self.error_recovery.classify_error(
                            e, component="screenshot", operation="capture"
                        )
                        recovered = self.error_recovery.handle_error(context, self)
                        
                        if recovered:
                            # Tentar novamente após recovery
                            return self.screenshot.capture()
                    
                    raise RuntimeError("Falha crítica na captura de screenshot")
    
    def _safe_detect_state(self, screenshot: np.ndarray) -> Tuple[str, float]:
        """Detecta estado com retry e recovery."""
        
        max_attempts = 2
        
        for attempt in range(max_attempts):
            try:
                if self.unified_detector:
                    result = self.unified_detector.detect(screenshot)
                    return result.state, result.confidence
                elif self.state_finder:
                    result = self.state_finder.find_state(screenshot)
                    return result, 0.7
                else:
                    return "unknown", 0.0
            
            except Exception as e:
                logger.warning(f"[WRAPPER] Falha na detecção ({attempt + 1}/{max_attempts}): {e}")
                
                if attempt < max_attempts - 1:
                    time.sleep(0.3)
                else:
                    if self.enable_error_recovery:
                        context = self.error_recovery.classify_error(
                            e, component="state_detector", operation="detect"
                        )
                        self.error_recovery.handle_error(context, self)
                    
                    return "unknown", 0.0
    
    def _safe_execute_state_handler(self, state: str, screenshot: np.ndarray):
        """Executa handler de estado com tratamento de erro."""
        
        try:
            handler = self.state_manager.states.get(state)
            if handler:
                if state in ["in_game", "unknown"]:
                    handler(screenshot)
                else:
                    handler()
        
        except Exception as e:
            logger.error(f"[WRAPPER] Erro ao executar handler para {state}: {e}")
            
            if self.enable_error_recovery:
                context = self.error_recovery.classify_error(
                    e, component="state_manager", operation=f"handle_{state}"
                )
                self.error_recovery.handle_error(context, self)
    
    def _safe_update_metrics(self):
        """Atualiza métricas sem propagar erros."""
        
        try:
            if self.observability:
                # Atualizar métricas
                pass
        except Exception as e:
            logger.debug(f"[WRAPPER] Erro ao atualizar métricas (não crítico): {e}")
    
    def _error_monitor_loop(self):
        """Loop de monitoramento de erros (thread separada)."""
        
        while self.running:
            try:
                if self.enable_error_recovery:
                    stats = self.error_recovery.get_stats()
                    
                    # Log se houver muitos erros
                    if stats["total_errors"] > 10:
                        logger.warning(
                            f"[ERROR_MONITOR] Muitos erros detectados: {stats['total_errors']}, "
                            f"recovery rate: {stats['recovery_rate']:.2%}"
                        )
                    
                    # Verificar circuit breakers
                    for component, state in stats.get("circuit_breakers", {}).items():
                        if state == "OPEN":
                            logger.warning(f"[ERROR_MONITOR] Circuit breaker ABERTO para {component}")
                
                time.sleep(30.0)  # Verificar a cada 30 segundos
            
            except Exception as e:
                logger.error(f"[ERROR_MONITOR] Erro no monitor loop: {e}")
                time.sleep(10.0)
    
    def get_error_recovery_stats(self) -> Dict:
        """Retorna estatísticas de error recovery."""
        
        if self.enable_error_recovery and hasattr(self, 'error_recovery'):
            return self.error_recovery.get_stats()
        
        return {"enabled": False}
    
    def stop(self):
        """Para o bot e sistemas de recovery."""
        
        logger.info("[WRAPPER] Parando bot...")
        
        self.running = False
        
        # Parar threads de monitoramento
        if hasattr(self, '_error_monitor_thread'):
            self._error_monitor_thread.join(timeout=2.0)
        
        # ... restante do stop existente ...
        
        logger.info("[WRAPPER] Bot parado")
```

---

## 📊 Monitoramento

### Verificar Estatísticas de Error Recovery

```python
# Em qualquer lugar do código
stats = bot.get_error_recovery_stats()
print(f"Total de erros: {stats['total_errors']}")
print(f"Taxa de recovery: {stats['recovery_rate']:.2%}")
print(f"Circuit breakers: {stats['circuit_breakers']}")
```

### Via Dashboard

Adicionar ao dashboard:

```python
# Em dashboard_server.py
def get_snapshot(self):
    with self._lock:
        data = asdict(self._data)
        
        # Adicionar stats de error recovery
        if wrapper_instance and hasattr(wrapper_instance, 'get_error_recovery_stats'):
            data["error_recovery"] = wrapper_instance.get_error_recovery_stats()
        
        return data
```

---

## 🎯 Benefícios

1. **Robustez**: Bot se recupera automaticamente de erros comuns
2. **Circuit Breakers**: Evita loops infinitos de falhas
3. **Logging**: Erros são classificados e logados detalhadamente
4. **Granularidade**: Tratamento específico por tipo de erro
5. **Estatísticas**: Monitoramento de saúde do sistema
6. **Non-intrusive**: Pode ser habilitado/desabilitado facilmente

---

## ⚙️ Configuração

### Habilitar/Desabilitar

```python
# No __init__ do wrapper
bot = PylaAIEnhanced(enable_error_recovery=True)  # Habilitar
bot = PylaAIEnhanced(enable_error_recovery=False) # Desabilitar
```

### Ajustar Parâmetros

```python
self.error_recovery = ErrorRecoverySystem(
    enable_auto_recovery=True,
    max_recovery_attempts=5,  # Aumentar tentativas
    global_circuit_breaker=True
)
```

---

## 🧪 Testes

### Testar Recovery Manual

```python
# Simular erro de screenshot
try:
    screenshot = bot.screenshot.capture()
except Exception as e:
    context = bot.error_recovery.classify_error(e, "screenshot", "capture")
    recovered = bot.error_recovery.handle_error(context, bot)
    print(f"Recovery: {'sucesso' if recovered else 'falhou'}")
```

---

## 📝 Notas

- O sistema é **backward compatible** - funciona com ou sem ele
- **Não substitui** try-catch existentes, mas complementa
- Use logging detalhado para entender o que está acontecendo
- Monitore as estatísticas regularmente
- Ajuste parâmetros baseado no comportamento observado
