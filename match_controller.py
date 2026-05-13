"""
match_controller.py

Controlo de partidas para Brawl Stars.
Inicia e termina partidas automaticamente, gere troféus e histórico.
"""

import time
import json
from typing import Optional, Dict, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class MatchResult:
    """Resultado de uma partida"""
    match_id: str
    timestamp: str
    game_mode: str
    brawler: str
    result: str  # "win", "loss", "draw"
    trophies_change: int
    duration_seconds: float
    kills: int
    damage_dealt: int
    powerups_collected: int
    star_player: bool
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BrawlerConfig:
    """Configuração de um brawler para farm"""
    name: str
    current_trophies: int
    target_trophies: int
    current_wins: int
    target_wins: int
    priority: int = 1  # 1-5, maior = mais prioritário
    enabled: bool = True


class MatchHistory:
    """Histórico de partidas jogadas"""
    
    def __init__(self, history_file: Path):
        self.history_file = history_file
        self.matches: List[MatchResult] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Carrega histórico do ficheiro"""
        if self.history_file.exists():
            try:
                with open(self.history_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.matches = [MatchResult(**m) for m in data.get("matches", [])]
            except Exception as e:
                logger.error(f"Erro ao carregar histórico: {e}")
    
    def save(self) -> None:
        """Guarda histórico no ficheiro"""
        try:
            self.history_file.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "last_updated": datetime.now().isoformat(),
                "total_matches": len(self.matches),
                "matches": [m.to_dict() for m in self.matches]
            }
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Erro ao guardar histórico: {e}")
    
    def add_match(self, result: MatchResult) -> None:
        """Adiciona uma partida ao histórico"""
        self.matches.append(result)
        # Manter apenas últimas 1000 partidas
        if len(self.matches) > 1000:
            self.matches = self.matches[-1000:]
        self.save()
    
    def get_stats(self, last_n: Optional[int] = None) -> Dict:
        """Retorna estatísticas do histórico"""
        matches = self.matches[-last_n:] if last_n else self.matches
        
        if not matches:
            return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0}
        
        wins = sum(1 for m in matches if m.result == "win")
        losses = sum(1 for m in matches if m.result == "loss")
        
        return {
            "total": len(matches),
            "wins": wins,
            "losses": losses,
            "draws": sum(1 for m in matches if m.result == "draw"),
            "win_rate": (wins / len(matches)) * 100,
            "total_trophies": sum(m.trophies_change for m in matches),
            "avg_duration": sum(m.duration_seconds for m in matches) / len(matches),
            "total_kills": sum(m.kills for m in matches),
            "total_damage": sum(m.damage_dealt for m in matches)
        }


class BrawlerQueue:
    """Fila de brawlers para farm automático"""
    
    def __init__(self):
        self.brawlers: List[BrawlerConfig] = []
        self.current_index = 0
    
    def add_brawler(self, config: BrawlerConfig) -> None:
        """Adiciona brawler à fila"""
        self.brawlers.append(config)
        # Ordenar por prioridade
        self.brawlers.sort(key=lambda b: b.priority, reverse=True)
    
    def get_current(self) -> Optional[BrawlerConfig]:
        """Retorna brawler atual"""
        if not self.brawlers:
            return None
        return self.brawlers[self.current_index]
    
    def next(self) -> Optional[BrawlerConfig]:
        """Avança para o próximo brawler"""
        if not self.brawlers:
            return None
        
        self.current_index = (self.current_index + 1) % len(self.brawlers)
        return self.get_current()
    
    def should_switch(
        self,
        current_result: Optional[MatchResult] = None,
        history: Optional[MatchHistory] = None,
    ) -> bool:
        """Determina se deve trocar de brawler usando metas e histórico recente."""
        current = self.get_current()
        if not current:
            return False
        
        # Trocar se atingiu meta de troféus
        if current.current_trophies >= current.target_trophies:
            logger.info(f"Brawler {current.name} atingiu meta de troféus!")
            return True
        
        # Trocar se atingiu meta de vitórias
        if current.current_wins >= current.target_wins:
            logger.info(f"Brawler {current.name} atingiu meta de vitórias!")
            return True
        
        # Trocar após 3 derrotas seguidas
        recent_matches = history.matches[-3:] if history else []
        recent_losses = sum(
            1 for m in recent_matches
            if m.result == "loss" and m.brawler == current.name
        )
        if recent_losses >= 3:
            logger.info(f"Brawler {current.name} com 3 derrotas seguidas, trocando...")
            return True
        
        return False


class MatchController:
    """
    Controlador principal de partidas.
    Gerencia início/fim de partidas, troféus e troca de brawlers.
    """
    
    def __init__(self, install_path: Path):
        self.install_path = install_path
        self.history = MatchHistory(install_path / "match_history.json")
        self.brawler_queue = BrawlerQueue()
        
        self.current_match: Optional[Dict] = None
        self.match_start_time: Optional[float] = None
        self.is_in_match = False
        
        # Estado do jogo
        self.total_trophies = 0
        self.session_matches = 0
        self.session_start = time.time()
    
    def start_match(self, game_mode: str, brawler: str) -> bool:
        """Inicia uma nova partida"""
        logger.debug(f"[MATCH_CONTROLLER] start_match chamado: game_mode={game_mode}, brawler={brawler}")
        if self.is_in_match or self.current_match is not None:
            logger.warning("[MATCH_CONTROLLER] Já existe uma partida em andamento ou pendente de finalização!")
            logger.debug(f"[MATCH_CONTROLLER] Estado atual: is_in_match={self.is_in_match}, current_match={self.current_match}")
            return False
        
        self.current_match = {
            "id": f"match_{int(time.time())}",
            "game_mode": game_mode,
            "brawler": brawler,
            "start_time": datetime.now().isoformat()
        }
        self.match_start_time = time.time()
        self.is_in_match = True
        
        logger.info(f"[MATCH_CONTROLLER] 🎮 Partida iniciada: {game_mode} com {brawler}")
        logger.debug(f"[MATCH_CONTROLLER] Estado atualizado: is_in_match={self.is_in_match}, match_id={self.current_match['id']}")
        return True

    def reset_match(self) -> None:
        """Limpa explicitamente qualquer estado de partida pendente."""
        logger.debug("[MATCH_CONTROLLER] reset_match chamado")
        logger.debug(f"[MATCH_CONTROLLER] Estado antes do reset: is_in_match={self.is_in_match}, current_match={self.current_match}")
        self.current_match = None
        self.match_start_time = None
        self.is_in_match = False
        logger.debug("[MATCH_CONTROLLER] Estado após reset: is_in_match=False, current_match=None")

    def _advance_queue_if_needed(self, match_result: MatchResult) -> None:
        """Avança a fila somente quando o resultado confirma troca."""
        should_switch = self.brawler_queue.should_switch(match_result, history=self.history)
        if not should_switch:
            return

        next_brawler = self.brawler_queue.next()
        if next_brawler:
            logger.info(f"Fila avançada para {next_brawler.name} após {match_result.result}")
        else:
            logger.info("Fila não avançou porque não há próximo brawler ativo")
    
    def end_match(self, result: str, kills: int = 0, damage: int = 0, 
                  powerups: int = 0, star_player: bool = False) -> MatchResult:
        """Termina a partida atual"""
        logger.debug(f"[MATCH_CONTROLLER] end_match chamado: result={result}, kills={kills}")
        if not self.is_in_match or self.current_match is None:
            logger.warning("[MATCH_CONTROLLER] Nenhuma partida em andamento para encerrar!")
            logger.debug(f"[MATCH_CONTROLLER] Estado atual: is_in_match={self.is_in_match}, current_match={self.current_match}")
            return None
        
        duration = time.time() - self.match_start_time if self.match_start_time else 0
        
        # Calcular mudança de troféus
        trophies_change = self._calculate_trophy_change(result, star_player)
        
        match_result = MatchResult(
            match_id=self.current_match["id"],
            timestamp=datetime.now().isoformat(),
            game_mode=self.current_match["game_mode"],
            brawler=self.current_match["brawler"],
            result=result,
            trophies_change=trophies_change,
            duration_seconds=duration,
            kills=kills,
            damage_dealt=damage,
            powerups_collected=powerups,
            star_player=star_player
        )
        
        # Guardar no histórico
        self.history.add_match(match_result)
        
        # Atualizar estatísticas
        self.total_trophies += trophies_change
        self.session_matches += 1
        
        # Atualizar brawler atual
        current_brawler = self.brawler_queue.get_current()
        if current_brawler and current_brawler.name == match_result.brawler:
            current_brawler.current_trophies += trophies_change
            if result == "win":
                current_brawler.current_wins += 1

        self._advance_queue_if_needed(match_result)
        
        self.reset_match()
        
        logger.info(f"🏁 Partida terminada: {result} | Troféus: {trophies_change:+d} | Duração: {duration:.1f}s")
        
        return match_result
    
    def _calculate_trophy_change(self, result: str, star_player: bool) -> int:
        """Calcula mudança de troféus baseado no resultado"""
        base_change = {
            "win": 8,
            "loss": -6,
            "draw": 0
        }.get(result, 0)
        
        # Bónus star player
        if star_player and result == "win":
            base_change += 2
        
        return base_change
    
    def get_recommended_action(self) -> str:
        """Recomenda ação baseada no estado atual"""
        if not self.is_in_match:
            return "start_match"
        
        # Verificar se deve trocar de brawler
        if self.brawler_queue.should_switch(history=self.history):
            return "switch_brawler"
        
        return "continue"

    def get_session_info(self) -> Dict:
        """Retorna informações da sessão atual para monitoramento."""
        return {
            "total_trophies": self.total_trophies,
            "session_matches": self.session_matches,
            "is_in_match": self.is_in_match,
            "current_brawler": self.brawler_queue.get_current().name if self.brawler_queue.get_current() else None,
            "session_duration_seconds": time.time() - self.session_start,
        }
    
    def get_session_stats(self) -> Dict:
        """Retorna estatísticas da sessão atual"""
        session_duration = time.time() - self.session_start

        return {
            "duration_minutes": session_duration / 60,
            "matches_played": self.session_matches,
            "trophies_gained": self.total_trophies,
            "total_trophies": self.total_trophies,
            "session_matches": self.session_matches,
            "is_in_match": self.is_in_match,
            "avg_match_duration": session_duration / max(1, self.session_matches) / 60,
            "current_brawler": self.brawler_queue.get_current().name if self.brawler_queue.get_current() else None,
            "history": self.history.get_stats(last_n=10)
        }
    
    def auto_queue_brawlers(self, brawler_list: List[Dict]) -> None:
        """Configura fila automática de brawlers"""
        for brawler_data in brawler_list:
            config = BrawlerConfig(
                name=brawler_data["name"],
                current_trophies=brawler_data.get("trophies", 0),
                target_trophies=brawler_data.get("target_trophies", 350),
                current_wins=brawler_data.get("wins", 0),
                target_wins=brawler_data.get("target_wins", 10),
                priority=brawler_data.get("priority", 1),
                enabled=brawler_data.get("enabled", True)
            )
            if config.enabled:
                self.brawler_queue.add_brawler(config)
        
        logger.info(f"🎯 Fila de brawlers configurada: {len(self.brawler_queue.brawlers)} brawlers")


class AutoPicker:
    """Seleciona brawlers automaticamente baseado no mapa e modo"""
    
    def __init__(self):
        # Meta brawlers por modo
        self.mode_recommendations = {
            "Gem Grab": ["Gene", "Tara", "Sprout", "Poco", "Rosa"],
            "Brawl Ball": ["Rosa", "El Primo", "Jacky", "Bibi", "Fang"],
            "Heist": ["Colt", "Rico", "Brock", "Dynamike", "8-Bit"],
            "Bounty": ["Tick", "Piper", "Brock", "Byron", "Crow"],
            "Siege": ["Rosa", "Jacky", "Pam", "8-Bit", "Jessie"],
            "Hot Zone": ["Rosa", "Emz", "Poco", "Sandy", "Gale"],
            "Showdown": ["Shelly", "El Primo", "Bull", "Crow", "Leon"]
        }
    
    def recommend_brawler(self, game_mode: str, available: List[str]) -> Optional[str]:
        """Recomenda melhor brawler disponível para o modo"""
        recommendations = self.mode_recommendations.get(game_mode, [])
        
        for brawler in recommendations:
            if brawler in available:
                return brawler
        
        # Fallback: primeiro disponível
        return available[0] if available else None
    
    def analyze_team_comp(self, team_brawlers: List[str]) -> Dict:
        """Analisa composição da equipa"""
        roles = {
            "tank": ["El Primo", "Rosa", "Bull", "Jacky", "Frank", "Bibi"],
            "damage": ["Colt", "Rico", "Brock", "Dynamike", "8-Bit", "Carl"],
            "support": ["Poco", "Pam", "Gene", "Max", "Sandy", "Byron"],
            "assassin": ["Crow", "Leon", "Mortis", "Stu", "Fang"],
            "control": ["Sprout", "Tara", "Emz", "Gale", "Surge"]
        }
        
        team_roles = []
        for brawler in team_brawlers:
            for role, brawlers in roles.items():
                if brawler in brawlers:
                    team_roles.append(role)
                    break
        
        return {
            "roles": team_roles,
            "has_tank": "tank" in team_roles,
            "has_support": "support" in team_roles,
            "has_damage": "damage" in team_roles,
            "balanced": len(set(team_roles)) >= 2
        }
