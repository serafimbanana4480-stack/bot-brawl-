"""
humanization.py

Sistema avançado de humanização para o bot de Brawl Stars.
Movimentos de rato humanizados com curvas de Bézier, delays aleatórios,
e comportamento variável para evitar deteção.
"""

import random
import time
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np


@dataclass
class HumanizationConfig:
    """Configurações de humanização"""
    enabled: bool = True
    min_delay: float = 0.3
    max_delay: float = 1.5
    bezier_control_variation: float = 0.3
    tremor_amplitude: float = 2.0
    tremor_frequency: float = 0.1
    mistake_probability: float = 0.1
    reaction_time_base: float = 0.25
    reaction_time_variance: float = 0.15


class BezierCurve:
    """Curva de Bézier para movimentos suaves do rato"""
    
    def __init__(self, p0: Tuple[float, float], p1: Tuple[float, float], 
                 p2: Tuple[float, float], p3: Tuple[float, float]):
        self.p0 = np.array(p0)
        self.p1 = np.array(p1)
        self.p2 = np.array(p2)
        self.p3 = np.array(p3)
    
    def get_point(self, t: float) -> Tuple[float, float]:
        """Calcula ponto na curva para t em [0, 1]"""
        t = max(0.0, min(1.0, t))
        
        # Cubic Bezier formula
        p = (1-t)**3 * self.p0 + \
            3*(1-t)**2*t * self.p1 + \
            3*(1-t)*t**2 * self.p2 + \
            t**3 * self.p3
        
        return (float(p[0]), float(p[1]))
    
    def generate_path(self, num_points: int = 50) -> List[Tuple[float, float]]:
        """Gera caminho completo da curva"""
        return [self.get_point(t / (num_points - 1)) for t in range(num_points)]


class WindMouse:
    """
    WindMouse Algorithm - Movimento de mouse humanizado mais realista.
    Baseado no algoritmo usado em bots profissionais.
    """
    
    def __init__(self, gravity: float = 9.0, wind: float = 3.0):
        """
        Args:
            gravity: Força que puxa o mouse em direção ao destino
            wind: Força aleatória que adiciona "jitter" ao movimento
        """
        self.gravity = gravity
        self.wind = wind
    
    def generate_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        max_step: int = 20
    ) -> List[Tuple[float, float]]:
        """
        Gera caminho usando WindMouse algorithm.
        
        Args:
            start: Ponto inicial (x, y)
            end: Ponto final (x, y)
            max_step: Distância máxima por passo
        
        Returns:
            Lista de pontos (x, y)
        """
        path = []
        current_x, current_y = start
        target_x, target_y = end
        
        while True:
            # Calcular distância até o destino
            dist = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
            
            if dist < max_step:
                path.append((target_x, target_y))
                break
            
            # Calcular vetor de direção
            vx = (target_x - current_x) / dist
            vy = (target_y - current_y) / dist
            
            # Adicionar wind (jitter aleatório)
            wind_x = random.uniform(-self.wind, self.wind)
            wind_y = random.uniform(-self.wind, self.wind)
            
            # Adicionar gravity (puxa em direção ao destino)
            gravity_x = vx * self.gravity
            gravity_y = vy * self.gravity
            
            # Calcular novo ponto
            new_x = current_x + vx * max_step + wind_x + gravity_x
            new_y = current_y + vy * max_step + wind_y + gravity_y
            
            # Limitar para não ultrapassar o destino (funciona em qualquer direção)
            new_x = min(max(new_x, min(current_x, target_x)), max(current_x, target_x))
            new_y = min(max(new_y, min(current_y, target_y)), max(current_y, target_y))
            
            path.append((new_x, new_y))
            current_x, current_y = new_x, new_y
        
        return path
    
    def generate_path_with_timing(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        base_speed: float = 500.0
    ) -> List[Tuple[float, float, float]]:
        """
        Gera caminho com timestamps para simular velocidade variável.
        
        Returns:
            Lista de (x, y, timestamp)
        """
        points = self.generate_path(start, end)
        
        # Calcular distância total
        total_dist = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            total_dist += math.sqrt(dx**2 + dy**2)
        
        # Calcular tempo total baseado na velocidade
        total_time = total_dist / base_speed
        
        # Distribuir tempo com easing (mais lento no início e fim)
        path_with_timing = []
        current_time = 0.0
        
        for i, (x, y) in enumerate(points):
            t = i / (len(points) - 1) if len(points) > 1 else 0
            
            # Ease-in-out cubic
            t_eased = 4 * t**3 if t < 0.5 else 1 - ((-2 * t + 2)**3) / 2
            
            timestamp = total_time * t_eased
            path_with_timing.append((x, y, timestamp))
        
        return path_with_timing


class MouseHumanizer:
    """Humanizador de movimentos de rato"""
    
    def __init__(self, config: Optional[HumanizationConfig] = None):
        self.config = config or HumanizationConfig()
        self.windmouse = WindMouse()
    
    def generate_bezier_control_points(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        variation: Optional[float] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Gera pontos de controlo para curva de Bézier com variação natural.
        """
        if variation is None:
            variation = self.config.bezier_control_variation
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Pontos de controlo com offset aleatório
        # Control point 1: ~30% do caminho com offset perpendicular
        t1 = 0.3
        mid1 = (start[0] + dx * t1, start[1] + dy * t1)
        
        # Offset perpendicular aleatório
        perp_angle = math.atan2(dy, dx) + math.pi / 2
        offset1 = random.uniform(-distance * variation, distance * variation)
        
        cp1 = (
            mid1[0] + math.cos(perp_angle) * offset1,
            mid1[1] + math.sin(perp_angle) * offset1
        )
        
        # Control point 2: ~70% do caminho com offset perpendicular
        t2 = 0.7
        mid2 = (start[0] + dx * t2, start[1] + dy * t2)
        offset2 = random.uniform(-distance * variation, distance * variation)
        
        cp2 = (
            mid2[0] + math.cos(perp_angle) * offset2,
            mid2[1] + math.sin(perp_angle) * offset2
        )
        
        return cp1, cp2
    
    def humanize_path(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        duration: Optional[float] = None,
        use_windmouse: bool = True
    ) -> List[Tuple[float, float, float]]:
        """
        Gera caminho humanizado com timestamps.
        
        Args:
            start: Ponto inicial
            end: Ponto final
            duration: Duração total (opcional, calculada automaticamente se None)
            use_windmouse: Se True, usa WindMouse (mais realista). Se False, usa Bézier.
        
        Returns:
            [(x, y, timestamp), ...]
        """
        if use_windmouse:
            # Usar WindMouse (mais realista)
            if duration is None:
                distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                speed = random.uniform(500, 1500)
                duration = distance / speed
            
            path_with_time = self.windmouse.generate_path_with_timing(start, end, base_speed=1000.0)
            
            # Adicionar tremor sutil
            path_with_tremor = []
            for x, y, timestamp in path_with_time:
                tremor_x = random.gauss(0, self.config.tremor_amplitude)
                tremor_y = random.gauss(0, self.config.tremor_amplitude)
                path_with_tremor.append((x + tremor_x, y + tremor_y, timestamp))
            
            return path_with_tremor
        else:
            # Usar Bézier (método original)
            if duration is None:
                distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
                speed = random.uniform(500, 1500)
                duration = distance / speed
            
            cp1, cp2 = self.generate_bezier_control_points(start, end)
            curve = BezierCurve(start, cp1, cp2, end)
            
            num_points = max(20, int(duration * 60))
            
            path_with_time = []
            for i in range(num_points):
                t = i / (num_points - 1)
                t_eased = self._ease_in_out_cubic(t)
                point = curve.get_point(t_eased)
                timestamp = duration * t
                
                tremor_x = random.gauss(0, self.config.tremor_amplitude)
                tremor_y = random.gauss(0, self.config.tremor_amplitude)
                
                path_with_time.append((
                    point[0] + tremor_x,
                    point[1] + tremor_y,
                    timestamp
                ))
            
            return path_with_time
    
    def _ease_in_out_cubic(self, t: float) -> float:
        """Função de easing suave"""
        if t < 0.5:
            return 4 * t * t * t
        else:
            f = 2 * t - 2
            return 1 + 0.5 * f * f * f
    
    def should_make_mistake(self) -> bool:
        """Decide se deve "errar" intencionalmente"""
        return random.random() < self.config.mistake_probability
    
    def get_mistake_offset(self, max_offset: float = 50.0) -> Tuple[float, float]:
        """Gera offset para "erro" de mira"""
        angle = random.uniform(0, 2 * math.pi)
        distance = random.uniform(10, max_offset)
        return (math.cos(angle) * distance, math.sin(angle) * distance)


class DelayRandomizer:
    """Randomizador de delays entre ações"""
    
    def __init__(self, config: Optional[HumanizationConfig] = None):
        self.config = config or HumanizationConfig()
    
    def get_delay(self, action_type: str = "default") -> float:
        """
        Gera delay aleatório com distribuição gaussiana.
        action_types: "default", "reaction", "decision", "movement"
        """
        if not self.config.enabled:
            return 0.0
        action_type = (action_type or "default").lower()
        profiles = {
            "reaction": (self.config.reaction_time_base, self.config.reaction_time_variance),
            "decision": (0.55, 0.30),
            "movement": (0.18, 0.08),
            "tap": (0.20, 0.12),
            "attack": (0.16, 0.10),
            "super": (0.28, 0.14),
            "menu": (0.32, 0.18),
            "default": ((self.config.min_delay + self.config.max_delay) / 2, (self.config.max_delay - self.config.min_delay) / 4),
        }
        base, variance = profiles.get(action_type, profiles["default"])
        
        delay = random.gauss(base, variance)
        
        # Clamp aos limites
        return max(self.config.min_delay, min(self.config.max_delay, delay))
    
    def get_typing_delay(self, char: str = "") -> float:
        """Delay para simular digitação"""
        # Caracteres comuns: 80-150ms
        # Caracteres especiais: 150-250ms
        if char in "abcdefghijklmnopqrstuvwxyz0123456789":
            return random.uniform(0.08, 0.15)
        else:
            return random.uniform(0.15, 0.25)
    
    def sleep(self, action_type: str = "default") -> None:
        """Executa sleep com delay aleatório"""
        time.sleep(self.get_delay(action_type))


class BehaviorRandomizer:
    """Randomizador de comportamento para evitar padrões"""
    
    def __init__(self):
        self.pattern_history: List[dict] = []
        self.last_direction_change = time.time()
        self.current_bias = {"x": 0, "y": 0}
    
    def should_change_pattern(self) -> bool:
        """Decide se deve mudar padrão de comportamento"""
        # Mudar padrão a cada 30-120 segundos
        elapsed = time.time() - self.last_direction_change
        return elapsed > random.uniform(30, 120)
    
    def get_random_bias(self) -> Tuple[float, float]:
        """Gera bias aleatório para movimentos"""
        if self.should_change_pattern():
            self.current_bias = {
                "x": random.uniform(-0.3, 0.3),
                "y": random.uniform(-0.3, 0.3)
            }
            self.last_direction_change = time.time()
        
        return (self.current_bias["x"], self.current_bias["y"])
    
    def get_random_action_priority(self, actions: List[str]) -> List[str]:
        """Randomiza prioridade de ações"""
        weights = [random.uniform(0.5, 1.5) for _ in actions]
        total = sum(weights)
        probs = [w/total for w in weights]
        
        # Shuffle baseado nas probabilidades
        result = actions.copy()
        for i in range(len(result) - 1, 0, -1):
            if random.random() < probs[i]:
                j = random.randint(0, i)
                result[i], result[j] = result[j], result[i]
        
        return result


class HumanizationEngine:
    """Motor principal de humanização"""
    
    def __init__(self, config: Optional[HumanizationConfig] = None):
        self.config = config or HumanizationConfig()
        self.mouse = MouseHumanizer(self.config)
        self.delays = DelayRandomizer(self.config)
        self.behavior = BehaviorRandomizer()
        
        self.action_count = 0
        self.last_pause = time.time()
        self.pause_interval = random.uniform(300, 600)  # 5-10 minutos
    
    def should_take_break(self) -> bool:
        """Verifica se deve fazer uma pausa"""
        elapsed = time.time() - self.last_pause
        return elapsed > self.pause_interval
    
    def get_break_duration(self) -> float:
        """Retorna duração da pausa"""
        self.last_pause = time.time()
        self.pause_interval = random.uniform(300, 600)
        return random.uniform(30, 60)  # 30-60 segundos
    
    def execute_humanized_click(
        self, 
        x: float, 
        y: float, 
        pre_delay: Optional[float] = None
    ) -> dict:
        """
        Executa clique humanizado.
        Retorna informações sobre o movimento.
        """
        # Delay pré-acção
        if pre_delay is None:
            pre_delay = self.delays.get_delay("reaction")
        time.sleep(pre_delay)
        
        # Verificar se deve "errar"
        is_mistake = False
        if self.mouse.should_make_mistake():
            offset = self.mouse.get_mistake_offset()
            x += offset[0]
            y += offset[1]
            is_mistake = True
        
        # Gerar caminho (não executamos aqui, só retornamos para o executor)
        # Assumindo posição atual do rato como (0, 0) para exemplo
        path = self.mouse.humanize_path((0, 0), (x, y))
        
        self.action_count += 1
        
        return {
            "target": (x, y),
            "path": path,
            "pre_delay": pre_delay,
            "is_mistake": is_mistake,
            "timestamp": time.time(),
            "action_number": self.action_count
        }
    
    def get_humanized_aim(
        self, 
        target: Tuple[float, float],
        accuracy: float = 0.9
    ) -> Tuple[float, float]:
        """
        Retorna coordenadas de mira humanizadas.
        accuracy: 1.0 = perfeito, 0.0 = totalmente errado
        """
        if random.random() > accuracy:
            # Miss
            offset = self.mouse.get_mistake_offset(max_offset=100)
            return (target[0] + offset[0], target[1] + offset[1])
        
        return target
    
    def get_stats(self) -> dict:
        """Retorna estatísticas de humanização"""
        return {
            "total_actions": self.action_count,
            "time_since_last_pause": time.time() - self.last_pause,
            "next_pause_in": self.pause_interval - (time.time() - self.last_pause),
            "config": {
                "min_delay": self.config.min_delay,
                "max_delay": self.config.max_delay,
                "mistake_probability": self.config.mistake_probability
            }
        }

    def get_action_delay(self) -> float:
        """Retorna delay para uma ação (usado pelo wrapper)"""
        return self.delays.get_delay("default")

    def get_delay(self, action_type: str = "default") -> float:
        """Public API: delay humanizado por tipo de ação.

        action_type: 'default' | 'reaction' | 'decision' | 'movement'
        """
        return self.delays.get_delay(action_type)

    def get_tremor(self) -> Tuple[float, float]:
        """Public API: retorna offset de tremor de rato (x, y) em píxeis.

        Usa amplitude configurada em HumanizationConfig.tremor_amplitude.
        """
        import random as _r
        amp = self.config.tremor_amplitude
        return (_r.gauss(0, amp), _r.gauss(0, amp))

    def get_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        use_windmouse: bool = True
    ) -> List[Tuple[float, float, float]]:
        """Public API: retorna caminho humanizado como lista de (x, y, timestamp).

        Args:
            start: posição inicial (x, y)
            end: posição final (x, y)
            use_windmouse: True usa WindMouse, False usa Bézier
        Returns:
            list of (x, y, timestamp_seconds)
        """
        return self.mouse.humanize_path(start, end, use_windmouse=use_windmouse)

    def move_mouse_curve(self, x1: float, y1: float, x2: float, y2: float) -> None:
        """Move o rato com curva Bézier (usado pelo wrapper)"""
        import pyautogui
        import time

        # Gerar caminho humanizado
        path = self.mouse.humanize_path((x1, y1), (x2, y2))

        # Executar movimento
        start = time.time()
        for x, y, timestamp in path:
            pyautogui.moveTo(x, y)
            target_elapsed = max(0.0, timestamp)
            current_elapsed = time.time() - start
            sleep_time = max(0.008, min(0.028, target_elapsed - current_elapsed + random.uniform(-0.004, 0.004)))
            time.sleep(sleep_time)


# Singleton instance para uso global
_default_engine: Optional[HumanizationEngine] = None


def get_humanization_engine(config: Optional[HumanizationConfig] = None) -> HumanizationEngine:
    """Retorna instância singleton do motor de humanização"""
    global _default_engine
    if _default_engine is None:
        _default_engine = HumanizationEngine(config)
    return _default_engine


class FatigueSimulator:
    """
    Simula fadiga ao longo da sessão.
    
    Aumenta delays, erros, e reduz precisão conforme a sessão progride.
    """
    
    def __init__(self, session_duration_hours: float = 3.0):
        self.session_duration = session_duration_hours * 3600  # Convert to seconds
        self.start_time = time.time()
        self.fatigue_level = 0.0  # 0.0 to 1.0
        
    def update_fatigue(self) -> float:
        """Atualiza nível de fadiga baseado no tempo de sessão"""
        elapsed = time.time() - self.start_time
        self.fatigue_level = min(1.0, elapsed / self.session_duration)
        return self.fatigue_level
    
    def get_fatigue_multiplier(self, metric: str) -> float:
        """
        Retorna multiplicador baseado na fadiga.
        
        Args:
            metric: "delay", "error_rate", "reaction_time"
        
        Returns:
            Multiplicador (1.0 = normal, >1.0 = aumentado)
        """
        fatigue = self.update_fatigue()
        
        if metric == "delay":
            # Delays aumentam até 2x com fadiga
            return 1.0 + fatigue
        elif metric == "error_rate":
            # Erros aumentam até 3x com fadiga
            return 1.0 + fatigue * 2.0
        elif metric == "reaction_time":
            # Tempo de reação aumenta até 1.5x com fadiga
            return 1.0 + fatigue * 0.5
        else:
            return 1.0
    
    def should_take_fatigue_break(self) -> bool:
        """Decide se deve fazer pausa devido à fadiga"""
        # Pausa recomendada se fadiga > 70%
        return self.fatigue_level > 0.7
    
    def get_fatigue_break_duration(self) -> float:
        """Retorna duração de pausa para recuperar fadiga"""
        # Pausa de 5-15 minutos para recuperar
        return random.uniform(300, 900)
    
    def recover_fatigue(self, break_duration: float):
        """Recupera fadiga após pausa"""
        recovery_factor = min(1.0, break_duration / 600.0)  # 10 min = recuperação completa
        self.fatigue_level *= (1.0 - recovery_factor * 0.8)
        self.start_time = time.time() - (self.fatigue_level * self.session_duration)


class PersonalityProfile:
    """
    Perfil de personalidade único por sessão.
    
    Define características comportamentais únicas para cada sessão
    para evitar padrões detectáveis.
    """
    
    def __init__(self):
        self.profile_id = self._generate_profile_id()
        self.aggression_level = random.uniform(0.3, 0.9)  # 0.0 = passivo, 1.0 = agressivo
        self.caution_level = random.uniform(0.2, 0.8)  # 0.0 = imprudente, 1.0 = cauteloso
        self.speed_preference = random.uniform(0.7, 1.3)  # Multiplicador de velocidade
        self.accuracy_preference = random.uniform(0.8, 1.0)  # Multiplicador de precisão
        self.risk_tolerance = random.uniform(0.3, 0.9)  # 0.0 = conservador, 1.0 = arriscado
        
        # Estilo de jogo
        self.playstyle = self._determine_playstyle()
        
    def _generate_profile_id(self) -> str:
        """Gera ID único para o perfil"""
        import uuid
        return str(uuid.uuid4())[:8]
    
    def _determine_playstyle(self) -> str:
        """Determina estilo de jogo baseado nos atributos"""
        if self.aggression_level > 0.7 and self.risk_tolerance > 0.7:
            return "aggressive"
        elif self.caution_level > 0.7:
            return "defensive"
        elif self.speed_preference > 1.1:
            return "rush"
        else:
            return "balanced"
    
    def get_adjusted_parameter(self, parameter: str, base_value: float) -> float:
        """
        Retorna parâmetro ajustado baseado no perfil.
        
        Args:
            parameter: "speed", "accuracy", "delay", "aggression"
            base_value: Valor base do parâmetro
        
        Returns:
            Valor ajustado
        """
        if parameter == "speed":
            return base_value * self.speed_preference
        elif parameter == "accuracy":
            return base_value * self.accuracy_preference
        elif parameter == "delay":
            # Menor agressividade = mais delay
            aggression_factor = 1.0 - (self.aggression_level - 0.5) * 0.3
            return base_value * aggression_factor
        elif parameter == "aggression":
            return base_value * self.aggression_level
        else:
            return base_value
    
    def should_engage(self, distance: float, enemy_count: int) -> bool:
        """
        Decide se deve engajar combate baseado no perfil.
        
        Args:
            distance: Distância ao inimigo
            enemy_count: Número de inimigos
        
        Returns:
            True se deve engajar
        """
        # Perfis cautelosos evitam combate
        if self.caution_level > 0.7 and enemy_count > 1:
            return False
        
        # Perfis agressivos engajam mais
        if self.aggression_level > 0.7 and distance < 300:
            return True
        
        # Decisão baseada em risco
        risk_score = (distance / 500.0) + (enemy_count * 0.2)
        return risk_score < self.risk_tolerance
    
    def get_profile_summary(self) -> Dict:
        """Retorna resumo do perfil"""
        return {
            "profile_id": self.profile_id,
            "playstyle": self.playstyle,
            "aggression_level": self.aggression_level,
            "caution_level": self.caution_level,
            "speed_preference": self.speed_preference,
            "accuracy_preference": self.accuracy_preference,
            "risk_tolerance": self.risk_tolerance
        }


class AdvancedHumanizationEngine(HumanizationEngine):
    """
    Motor de humanização avançado com simulação de fadiga e perfis de personalidade.
    
    Estende HumanizationEngine com recursos adicionais de realismo.
    """
    
    def __init__(self, config: Optional[HumanizationConfig] = None, 
                 session_duration_hours: float = 3.0):
        super().__init__(config)
        self.fatigue = FatigueSimulator(session_duration_hours)
        self.personality = PersonalityProfile()
        
    def get_delay(self, action_type: str = "default") -> float:
        """Retorna delay com ajustes de fadiga e personalidade"""
        base_delay = self.delays.get_delay(action_type)
        
        # Aplica multiplicador de fadiga
        fatigue_mult = self.fatigue.get_fatigue_multiplier("delay")
        
        # Aplica ajuste de personalidade
        personality_delay = self.personality.get_adjusted_parameter(
            "delay", base_delay * fatigue_mult
        )
        
        return personality_delay
    
    def should_make_mistake(self) -> bool:
        """Decide se deve errar considerando fadiga"""
        base_prob = self.config.mistake_probability
        
        # Aumenta probabilidade com fadiga
        fatigue_mult = self.fatigue.get_fatigue_multiplier("error_rate")
        adjusted_prob = base_prob * fatigue_mult
        
        return random.random() < adjusted_prob
    
    def get_humanized_aim(self, target: Tuple[float, float], 
                        accuracy: float = 0.9) -> Tuple[float, float]:
        """Retorna mira com ajustes de personalidade"""
        # Aplica preferência de precisão do perfil
        adjusted_accuracy = accuracy * self.personality.accuracy_preference
        
        return super().get_humanized_aim(target, adjusted_accuracy)
    
    def should_take_break(self) -> bool:
        """Verifica se deve fazer pausa (fadiga ou intervalo normal)"""
        # Pausa por fadiga tem prioridade
        if self.fatigue.should_take_fatigue_break():
            return True
        
        # Pausa normal
        return super().should_take_break()
    
    def get_break_duration(self) -> float:
        """Retorna duração da pausa"""
        if self.fatigue.should_take_fatigue_break():
            duration = self.fatigue.get_fatigue_break_duration()
            self.fatigue.recover_fatigue(duration)
            return duration
        
        return super().get_break_duration()
    
    def get_stats(self) -> dict:
        """Retorna estatísticas com informações avançadas"""
        base_stats = super().get_stats()
        
        base_stats.update({
            "fatigue_level": self.fatigue.fatigue_level,
            "personality": self.personality.get_profile_summary(),
            "session_elapsed": time.time() - self.fatigue.start_time
        })
        
        return base_stats
