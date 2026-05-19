"""
behavioral_profile.py

Sistema de perfil comportamental para humanização avançada.
Gera perfis realistas que simulam jogadores humanos.

Funcionalidades:
- Perfil de sessão (idade, experiência, personalidade)
- Curva de aprendizagem (warmup, peak, fatigue)
- Micro-comportamentos (overcorrection, hesitation, tunnel vision)
- Padrões de sessão realistas
"""

import random
import time
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, List

logger = logging.getLogger("behavioral")


class Personality(Enum):
    """Personalidades disponíveis."""
    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    BALANCED = "balanced"
    TACTICAL = "tactical"


class SkillLevel(Enum):
    """Níveis de skill."""
    NOVICE = "novice"       # 200-400ms reaction
    INTERMEDIATE = "intermediate"  # 150-250ms
    ADVANCED = "advanced"   # 100-180ms
    EXPERT = "expert"      # 60-120ms


@dataclass
class SessionProfile:
    """Perfil de uma sessão de jogo."""
    personality: Personality = Personality.BALANCED
    skill_level: SkillLevel = SkillLevel.INTERMEDIATE

    # Reaction time (ms) - baseado no skill
    base_reaction_time: float = 200.0
    reaction_variance: float = 50.0

    # Idade perceptual (afeta precisão)
    age_factor: float = 1.0

    # Experiência (afeta velocidade de decisão)
    experience: float = 0.5

    # Variância de performance
    performance_variance: float = 0.15

    # Sessão
    session_start: float = field(default_factory=time.time)
    session_length: float = 0.0

    # Tendências comportamentais
    overcorrection_chance: float = 0.1
    hesitation_chance: float = 0.15
    tunnel_vision_chance: float = 0.08

    def get_reaction_time(self) -> float:
        """Retorna tempo de reação atual baseado no skill e variância."""
        return self.base_reaction_time + random.gauss(0, self.reaction_variance)

    def get_session_phase(self) -> str:
        """Retorna a fase atual da sessão."""
        elapsed = time.time() - self.session_start
        self.session_length = elapsed

        if elapsed < 300:  # 0-5 min: warmup
            return "warmup"
        elif elapsed < 1200:  # 5-20 min: peak
            return "peak"
        elif elapsed < 2400:  # 20-40 min: decline
            return "decline"
        else:  # 40+ min: fatigue
            return "fatigue"

    def get_performance_modifier(self) -> float:
        """Retorna modificador de performance baseado na fase da sessão."""
        phase = self.get_session_phase()

        if phase == "warmup":
            return 0.7 + 0.3 * (self.session_length / 300)
        elif phase == "peak":
            return 1.0
        elif phase == "decline":
            return 1.0 - 0.15 * ((self.session_length - 1200) / 1200)
        else:  # fatigue
            return max(0.5, 0.85 - 0.01 * ((self.session_length - 2400) / 600))


class BehavioralProfileManager:
    """Gestor de perfis comportamentais."""

    def __init__(self):
        self._profiles: Dict[str, SessionProfile] = {}
        self._current_session: Optional[SessionProfile] = None
        self._session_start: float = time.time()

    def generate_session_profile(
        self,
        personality: Optional[Personality] = None,
        skill_level: Optional[SkillLevel] = None
    ) -> SessionProfile:
        """Gera um novo perfil de sessão aleatório ou especificado."""

        if personality is None:
            personality = random.choice(list(Personality))

        if skill_level is None:
            skill_level = random.choice(list(SkillLevel))

        # Configurar tempos de reação baseado no skill
        if skill_level == SkillLevel.NOVICE:
            base_reaction = random.uniform(250, 400)
            variance = 80
            experience = random.uniform(0.3, 0.5)
        elif skill_level == SkillLevel.INTERMEDIATE:
            base_reaction = random.uniform(150, 250)
            variance = 50
            experience = random.uniform(0.5, 0.7)
        elif skill_level == SkillLevel.ADVANCED:
            base_reaction = random.uniform(100, 180)
            variance = 35
            experience = random.uniform(0.7, 0.9)
        else:  # EXPERT
            base_reaction = random.uniform(60, 120)
            variance = 20
            experience = random.uniform(0.9, 1.0)

        profile = SessionProfile(
            personality=personality,
            skill_level=skill_level,
            base_reaction_time=base_reaction,
            reaction_variance=variance,
            age_factor=random.uniform(0.7, 1.0),
            experience=experience,
            performance_variance=random.uniform(0.1, 0.25),
            overcorrection_chance=random.uniform(0.05, 0.2),
            hesitation_chance=random.uniform(0.1, 0.25),
            tunnel_vision_chance=random.uniform(0.05, 0.15),
        )

        self._current_session = profile
        self._session_start = time.time()

        logger.info(f"[BEHAVIORAL] Novo perfil gerado: {personality.value}, {skill_level.value}")
        return profile

    def get_current_profile(self) -> Optional[SessionProfile]:
        """Retorna o perfil atual."""
        return self._current_session

    def should_show_hesitation(self) -> bool:
        """Determina se deve mostrar hesitação."""
        if not self._current_session:
            return False

        phase = self._current_session.get_session_phase()
        if phase == "warmup":
            base_prob = 0.3
        elif phase == "peak":
            base_prob = self._current_session.hesitation_chance
        elif phase == "decline":
            base_prob = 0.2
        else:  # fatigue
            base_prob = 0.35

        return random.random() < base_prob

    def should_overcorrect(self) -> bool:
        """Determina se deve overcorrect (corrigir mais do que necessário)."""
        if not self._current_session:
            return False
        return random.random() < self._current_session.overcorrection_chance

    def should_tunnel_vision(self) -> bool:
        """Determina se deve ter tunnel vision (focar num alvo só)."""
        if not self._current_session:
            return False
        return random.random() < self._current_session.tunnel_vision_chance

    def get_delay_modifier(self) -> float:
        """Retorna modificador de delay baseado no perfil."""
        if not self._current_session:
            return 1.0

        perf = self._current_session.get_performance_modifier()

        # Skill afeta delays
        if self._current_session.skill_level == SkillLevel.NOVICE:
            skill_mult = 1.5
        elif self._current_session.skill_level == SkillLevel.INTERMEDIATE:
            skill_mult = 1.2
        elif self._current_session.skill_level == SkillLevel.ADVANCED:
            skill_mult = 1.0
        else:  # EXPERT
            skill_mult = 0.9

        return perf * skill_mult

    def log_profile_stats(self):
        """Log estatísticas do perfil atual."""
        if not self._current_session:
            logger.debug("[BEHAVIORAL] Nenhum perfil ativo")
            return

        profile = self._current_session
        phase = profile.get_session_phase()
        perf = profile.get_performance_modifier()

        logger.info(
            f"[BEHAVIORAL] Perfil: {profile.personality.value}, "
            f"{profile.skill_level.value}, "
            f"Fase: {phase}, "
            f"Performance: {perf:.2f}, "
            f"Tempo: {profile.session_length:.0f}s"
        )