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
import logging

logger = logging.getLogger(__name__)


@dataclass
class HumanizationConfig:
    """Configurações de humanização"""
    enabled: bool = True
    min_delay: float = 0.15   # mínimo 150ms — limite humano realista
    max_delay: float = 2.0    # máximo 2s para pausas de pensamento
    bezier_control_variation: float = 0.3
    tremor_amplitude: float = 2.0
    tremor_frequency: float = 0.1
    mistake_probability: float = 0.1
    reaction_time_base: float = 0.25
    reaction_time_variance: float = 0.15
    # Novos parâmetros avançados
    advanced_humanization: bool = True
    pink_noise_enabled: bool = True
    overshoot_probability: float = 0.05
    micro_pause_probability: float = 0.08
    micro_pause_duration_base: float = 0.15
    fitts_a: float = 0.08  # intercepto Fitts (s)
    fitts_b: float = 0.12  # slope Fitts (s/bit)
    fitts_target_width: float = 40.0  # largura alvo padrão (px)
    jitter_proximity_factor: float = 2.5  # multiplicador jitter perto do alvo


class PinkNoiseGenerator:
    """
    Gerador de ruído pink (1/f) usando filtro IIR de Paul Kellet.
    
    Ruído pink é mais realista que gaussiano puro para movimentos humanos,
    pois humanos têm correlação temporal (movimentos suaves com variações
    de baixa frequência dominantes).
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        # 7-stage filter state (estado persistente para continuidade)
        self._state = [0.0] * 7
        self._white_prev = 0.0
    
    def next(self) -> float:
        """Retorna amostra de ruído pink (1/f) via soma de 3 frequências."""
        t = random.random() * 1000.0
        # Soma de 3 senos com decaimento 1/f — aproximação simples de pink noise
        return (
            math.sin(t * 0.1) * 0.50 +
            math.sin(t * 0.03) * 0.33 +
            math.sin(t * 0.01) * 0.17
        )
    
    def generate_sequence(self, n: int) -> np.ndarray:
        """Gera sequência de n amostras."""
        return np.array([self.next() for _ in range(n)])


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
    
    Melhorias anti-ban:
    - Ruído pink (1/f) em vez de uniforme/gaussiano puro
    - Jitter posição-dependente (mais perto do alvo)
    - Micro-pausas em movimentos longos
    - Overshoot humano ocasional
    - Fitts's Law para tempo de movimento
    """
    
    def __init__(self, gravity: float = 9.0, wind: float = 3.0):
        """
        Args:
            gravity: Força que puxa o mouse em direção ao destino
            wind: Força aleatória que adiciona "jitter" ao movimento
        """
        self.gravity = gravity
        self.wind = wind
        self._pink = PinkNoiseGenerator()
    
    def _position_dependent_jitter(self, distance_to_target: float, 
                                   total_distance: float,
                                   base_jitter: float) -> float:
        """
        Jitter aumenta nos últimos 10% do caminho (humanos hesitam ao acertar).
        """
        if total_distance <= 0:
            return base_jitter
        proximity = 1.0 - (distance_to_target / total_distance)
        # Mais jitter apenas nos últimos 10% do caminho
        factor = 1.0 + (max(0, proximity - 0.9) * 10.0) * 2.5
        return base_jitter * factor
    
    def fitts_law_time(self, distance: float, target_width: float = 40.0,
                       a: float = 0.08, b: float = 0.12) -> float:
        """
        Calcula tempo de movimento pela Lei de Fitts (segundos).
        MT = a + b * log2(D / W + 1)
        
        Args:
            distance: distância ao alvo (px)
            target_width: largura do alvo (px)
            a: intercepto (s)
            b: slope (s/bit)
        """
        if distance <= 0 or target_width <= 0:
            return a
        index_of_difficulty = math.log2(distance / target_width + 1.0)
        mt = a + b * index_of_difficulty
        # Adiciona variância humana (~15% CV)
        mt *= random.gauss(1.0, 0.15)
        return max(a, mt)
    
    def generate_path(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        max_step: int = 20,
        target_width: float = 40.0,
        fitts_a: float = 0.08,
        fitts_b: float = 0.12,
        enable_overshoot: bool = True,
        enable_micro_pauses: bool = True,
    ) -> List[Tuple[float, float]]:
        """
        Gera caminho usando WindMouse algorithm melhorado.
        
        Args:
            start: Ponto inicial (x, y)
            end: Ponto final (x, y)
            max_step: Distância máxima por passo
            target_width: Largura do alvo para Fitts's Law
            fitts_a: Intercepto Fitts
            fitts_b: Slope Fitts
            enable_overshoot: Se True, ocasionalmente passa do alvo e corrige
            enable_micro_pauses: Se True, adiciona micro-pausas em movimentos longos
        
        Returns:
            Lista de pontos (x, y)
        """
        path = []
        current_x, current_y = start
        target_x, target_y = end
        
        total_distance = math.sqrt((target_x - start[0])**2 + (target_y - start[1])**2)
        
        # Overshoot humano: ~5% das vezes, passa do target e corrige
        actual_target = (target_x, target_y)
        overshoot_active = False
        if enable_overshoot and random.random() < 0.05:
            overshoot_active = True
            overshoot_dist = random.uniform(15, 45)
            angle = math.atan2(target_y - start[1], target_x - start[0])
            actual_target = (
                target_x + math.cos(angle) * overshoot_dist,
                target_y + math.sin(angle) * overshoot_dist
            )
            logger.debug("[HUMANIZE] Overshoot ativado: target=(%.1f, %.1f) -> actual=(%.1f, %.1f)",
                         target_x, target_y, actual_target[0], actual_target[1])
        
        # Micro-pausas: em movimentos longos (>300px), ~8% chance de pausa a meio
        micro_pause_ms = 0.0
        if enable_micro_pauses and total_distance > 300 and random.random() < 0.08:
            micro_pause_ms = random.uniform(0.05, 0.15)  # 50-150ms
        
        step_count = 0
        while True:
            # Calcular distância até o destino atual (pode ser overshoot)
            dist = math.sqrt((actual_target[0] - current_x)**2 + (actual_target[1] - current_y)**2)
            
            if dist < max_step:
                path.append((actual_target[0], actual_target[1]))
                break
            
            # Calcular vetor de direção
            vx = (actual_target[0] - current_x) / dist
            vy = (actual_target[1] - current_y) / dist
            
            # Wind com ruído pink (1/f) - mais natural que uniforme
            wind_x = self._pink.next() * self.wind
            wind_y = self._pink.next() * self.wind
            
            # Jitter posição-dependente: mais jitter nos últimos 10%
            jitter_mult = self._position_dependent_jitter(dist, total_distance, 1.0)
            wind_x *= jitter_mult
            wind_y *= jitter_mult
            
            # Adicionar gravity (puxa em direção ao destino)
            gravity_x = vx * self.gravity
            gravity_y = vy * self.gravity
            
            # Calcular novo ponto
            new_x = current_x + vx * max_step + wind_x + gravity_x
            new_y = current_y + vy * max_step + wind_y + gravity_y
            
            # Limitar para não ultrapassar o destino (funciona em qualquer direção)
            new_x = min(max(new_x, min(current_x, actual_target[0])), max(current_x, actual_target[0]))
            new_y = min(max(new_y, min(current_y, actual_target[1])), max(current_y, actual_target[1]))
            
            path.append((new_x, new_y))
            current_x, current_y = new_x, new_y
            step_count += 1
        
        # Se houve overshoot, adicionar correção de volta ao target real
        if overshoot_active:
            # Correção rápida mas não instantânea
            correction_steps = max(3, int(random.gauss(6, 2)))
            for i in range(1, correction_steps + 1):
                t = i / correction_steps
                t_eased = t * t * (3 - 2 * t)  # smoothstep
                cx = actual_target[0] + (target_x - actual_target[0]) * t_eased
                cy = actual_target[1] + (target_y - actual_target[1]) * t_eased
                # Adicionar jitter pink na correção
                cx += self._pink.next() * self.wind * 0.5
                cy += self._pink.next() * self.wind * 0.5
                path.append((cx, cy))
        
        logger.debug(
            "[HUMANIZE] Path generated: steps=%d, overshoot=%s, distance=%.1f, "
            "fitts_time=%.3fs, micro_pause=%.3fs",
            len(path), overshoot_active, total_distance,
            self.fitts_law_time(total_distance, target_width, fitts_a, fitts_b),
            micro_pause_ms
        )
        
        return path
    
    def generate_path_with_timing(
        self,
        start: Tuple[float, float],
        end: Tuple[float, float],
        base_speed: float = 500.0,
        target_width: float = 40.0,
        fitts_a: float = 0.08,
        fitts_b: float = 0.12,
    ) -> List[Tuple[float, float, float]]:
        """
        Gera caminho com timestamps para simular velocidade variável.
        Usa Fitts's Law para tempo total de movimento.
        Adiciona micro-pausa de 50-150ms a meio do caminho em movimentos longos (>300px).
        
        Returns:
            Lista de (x, y, timestamp)
        """
        points = self.generate_path(start, end, target_width=target_width,
                                     fitts_a=fitts_a, fitts_b=fitts_b)
        
        # Micro-pausa probabilística para movimentos longos
        total_dist = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        micro_pause = random.uniform(0.05, 0.15) if (total_dist > 300 and random.random() < 0.08) else 0.0
        
        # Calcular distância total percorrida
        path_dist = 0.0
        for i in range(1, len(points)):
            dx = points[i][0] - points[i-1][0]
            dy = points[i][1] - points[i-1][1]
            path_dist += math.sqrt(dx**2 + dy**2)
        
        # Tempo total pela Lei de Fitts (mais realista que distância/velocidade)
        fitts_time = self.fitts_law_time(path_dist, target_width, fitts_a, fitts_b)
        # Fallback se Fitts der valor muito baixo
        speed_time = path_dist / base_speed if base_speed > 0 else 0.0
        total_time = max(fitts_time, speed_time * 0.3)
        
        # Distribuir tempo com easing (mais lento no início e fim)
        path_with_timing = []
        
        mid_idx = len(points) // 2
        for i, (x, y) in enumerate(points):
            t = i / (len(points) - 1) if len(points) > 1 else 0
            
            # Ease-in-out cubic
            t_eased = 4 * t**3 if t < 0.5 else 1 - ((-2 * t + 2)**3) / 2
            
            timestamp = total_time * t_eased
            # Micro-pausa a meio do caminho
            if i == mid_idx and micro_pause > 0:
                timestamp += micro_pause
            path_with_timing.append((x, y, timestamp))
        
        return path_with_timing


class MouseHumanizer:
    """Humanizador de movimentos de rato"""
    
    def __init__(self, config: Optional[HumanizationConfig] = None):
        self.config = config or HumanizationConfig()
        self.windmouse = WindMouse()
        self._pink = PinkNoiseGenerator()
    
    def generate_bezier_control_points(
        self, 
        start: Tuple[float, float], 
        end: Tuple[float, float],
        variation: Optional[float] = None
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Gera pontos de controlo para curva de Bézier com variação natural.
        Usa ruído pink para offsets mais realistas.
        """
        if variation is None:
            variation = self.config.bezier_control_variation
        
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = math.sqrt(dx**2 + dy**2)
        
        # Pontos de controlo com offset de ruído pink (mais natural)
        # Control point 1: ~30% do caminho com offset perpendicular
        t1 = 0.3
        mid1 = (start[0] + dx * t1, start[1] + dy * t1)
        
        # Offset perpendicular com ruído pink
        perp_angle = math.atan2(dy, dx) + math.pi / 2
        offset1 = self._pink.next() * distance * variation
        
        cp1 = (
            mid1[0] + math.cos(perp_angle) * offset1,
            mid1[1] + math.sin(perp_angle) * offset1
        )
        
        # Control point 2: ~70% do caminho com offset perpendicular
        t2 = 0.7
        mid2 = (start[0] + dx * t2, start[1] + dy * t2)
        offset2 = self._pink.next() * distance * variation
        
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
        distance = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2)
        
        if use_windmouse:
            # Usar WindMouse melhorado (mais realista)
            if duration is None:
                # Usar Fitts's Law para tempo de movimento
                duration = self.windmouse.fitts_law_time(
                    distance,
                    self.config.fitts_target_width,
                    self.config.fitts_a,
                    self.config.fitts_b
                )
            
            path_with_time = self.windmouse.generate_path_with_timing(
                start, end,
                target_width=self.config.fitts_target_width,
                fitts_a=self.config.fitts_a,
                fitts_b=self.config.fitts_b
            )
            
            # Adicionar tremor com ruído pink (mais natural que gaussiano puro)
            path_with_tremor = []
            for i, (x, y, timestamp) in enumerate(path_with_time):
                # Jitter posição-dependente: mais perto do alvo = mais tremor
                dist_to_end = math.sqrt((end[0]-x)**2 + (end[1]-y)**2)
                jitter_mult = 1.0 + ((1.0 - dist_to_end / max(distance, 1.0)) ** 2) * self.config.jitter_proximity_factor
                
                tremor_x = self._pink.next() * self.config.tremor_amplitude * jitter_mult
                tremor_y = self._pink.next() * self.config.tremor_amplitude * jitter_mult
                
                # Micro-pausa: inserir pequeno delay extra em pontos aleatórios
                extra_delay = 0.0
                if (self.config.advanced_humanization and 
                    distance > 300 and 
                    random.random() < self.config.micro_pause_probability):
                    extra_delay = random.gauss(
                        self.config.micro_pause_duration_base,
                        self.config.micro_pause_duration_base * 0.3
                    )
                
                path_with_tremor.append((x + tremor_x, y + tremor_y, timestamp + extra_delay))
            
            logger.debug(
                "[HUMANIZE] WindMouse path: dist=%.1f, duration=%.3fs, "
                "points=%d, pink_noise=%s",
                distance, duration, len(path_with_tremor),
                self.config.pink_noise_enabled
            )
            return path_with_tremor
        else:
            # Usar Bézier (método original melhorado)
            if duration is None:
                duration = self.windmouse.fitts_law_time(
                    distance,
                    self.config.fitts_target_width,
                    self.config.fitts_a,
                    self.config.fitts_b
                )
            
            cp1, cp2 = self.generate_bezier_control_points(start, end)
            curve = BezierCurve(start, cp1, cp2, end)
            
            num_points = max(20, int(duration * 60))
            
            path_with_time = []
            for i in range(num_points):
                t = i / (num_points - 1)
                t_eased = self._ease_in_out_cubic(t)
                point = curve.get_point(t_eased)
                timestamp = duration * t
                
                # Tremor com ruído pink
                dist_to_end = math.sqrt((end[0]-point[0])**2 + (end[1]-point[1])**2)
                jitter_mult = 1.0 + ((1.0 - dist_to_end / max(distance, 1.0)) ** 2) * self.config.jitter_proximity_factor
                
                tremor_x = self._pink.next() * self.config.tremor_amplitude * jitter_mult
                tremor_y = self._pink.next() * self.config.tremor_amplitude * jitter_mult
                
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
        Gera delay aleatório com distribuição realista.
        action_types: "default", "reaction", "decision", "movement"
        
        Reaction times mínimo: 150ms (limite humano fisiológico).
        """
        if not self.config.enabled:
            return 0.0
        action_type = (action_type or "default").lower()
        profiles = {
            # Mínimo 150ms para reaction (input lag + rede + processamento visual)
            "reaction": (0.22, 0.08),   # média ~220ms, std ~80ms -> range ~150-400ms
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
        
        # Clamp: reaction nunca abaixo de 150ms
        if action_type == "reaction":
            delay = max(0.15, delay)
        
        # Clamp aos limites globais
        delay = max(self.config.min_delay, min(self.config.max_delay, delay))
        
        logger.debug("[HUMANIZE] Delay %s: base=%.3f, result=%.3fs", action_type, base, delay)
        return delay
    
    def get_typing_delay(self, char: str = "") -> float:
        """Delay para simular digitação com padrões humanos realistas."""
        # Caracteres comuns: 80-150ms
        # Caracteres especiais: 150-250ms
        # Pausas ocasionais (pensamento): 300-800ms
        if char in "abcdefghijklmnopqrstuvwxyz0123456789":
            base = random.uniform(0.08, 0.15)
        elif char in " ":
            # Espaços ligeiramente mais rápidos
            base = random.uniform(0.07, 0.12)
        else:
            base = random.uniform(0.15, 0.25)
        
        # ~3% chance de pausa de "pensamento" entre palavras
        if char == " " and random.random() < 0.03:
            base += random.uniform(0.3, 0.8)
            logger.debug("[HUMANIZE] Typing think-pause: +%.3fs", base)
        
        return base
    
    def sleep(self, action_type: str = "default") -> None:
        """Executa sleep com delay aleatório"""
        delay = self.get_delay(action_type)
        time.sleep(delay)
        return delay


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
        
        # Logging de parâmetros para debug
        logger.info(
            "[HUMANIZE] Engine initialized: advanced=%s, pink_noise=%s, "
            "fitts_a=%.3f, fitts_b=%.3f, reaction_base=%.3f",
            self.config.advanced_humanization,
            self.config.pink_noise_enabled,
            self.config.fitts_a,
            self.config.fitts_b,
            self.config.reaction_time_base
        )
    
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
        # Delay pré-acção (mínimo 150ms para reaction)
        if pre_delay is None:
            pre_delay = self.delays.get_delay("reaction")
        else:
            pre_delay = max(0.15, pre_delay)  # enforce human minimum
        time.sleep(pre_delay)
        
        # Verificar se deve "errar"
        is_mistake = False
        if self.mouse.should_make_mistake():
            offset = self.mouse.get_mistake_offset()
            x += offset[0]
            y += offset[1]
            is_mistake = True
            logger.debug("[HUMANIZE] Mistake injected: offset=(%.1f, %.1f)", offset[0], offset[1])
        
        # Gerar caminho (não executamos aqui, só retornamos para o executor)
        # Assumindo posição atual do rato como (0, 0) para exemplo
        path = self.mouse.humanize_path((0, 0), (x, y))
        
        self.action_count += 1
        
        logger.debug(
            "[HUMANIZE] Click: target=(%.1f, %.1f), pre_delay=%.3fs, "
            "mistake=%s, path_points=%d",
            x, y, pre_delay, is_mistake, len(path)
        )
        
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
            logger.debug("[HUMANIZE] Aim miss: offset=(%.1f, %.1f)", offset[0], offset[1])
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
                "mistake_probability": self.config.mistake_probability,
                "advanced_humanization": self.config.advanced_humanization,
                "pink_noise_enabled": self.config.pink_noise_enabled,
                "fitts_a": self.config.fitts_a,
                "fitts_b": self.config.fitts_b,
                "overshoot_probability": self.config.overshoot_probability,
                "micro_pause_probability": self.config.micro_pause_probability,
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
        Agora com ruído pink se advanced_humanization estiver ativo.
        """
        amp = self.config.tremor_amplitude
        if self.config.advanced_humanization and self.config.pink_noise_enabled:
            # Usar ruído pink para tremor mais natural
            pink = PinkNoiseGenerator()
            return (pink.next() * amp, pink.next() * amp)
        return (random.gauss(0, amp), random.gauss(0, amp))

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
