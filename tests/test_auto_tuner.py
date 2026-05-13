"""
test_auto_tuner.py

Testes para o sistema de auto-tuning de parâmetros.
"""

import pytest
import time
from unittest.mock import Mock, MagicMock
from brawl_bot.auto_tuner import AutoTuner, TuningConfig
from brawl_bot.match_controller import MatchResult
from datetime import datetime


class TestAutoTuner:
    """Testes para AutoTuner"""
    
    @pytest.fixture
    def mock_match_controller(self):
        """Mock do match controller"""
        controller = Mock()
        controller.get_stats = Mock(return_value={
            "total": 20,
            "wins": 10,
            "losses": 8,
            "draws": 2,
            "win_rate": 50.0,
            "total_trophies": 100,
            "avg_duration": 120.0,
            "total_kills": 30,
            "total_damage": 15000
        })
        return controller
    
    @pytest.fixture
    def auto_tuner(self, mock_match_controller):
        """Fixture do auto-tuner"""
        config = TuningConfig(
            min_matches_for_tuning=5,
            tuning_interval_hours=0,  # Imediato para testes
            win_rate_target=0.6
        )
        return AutoTuner(mock_match_controller, config)
    
    @pytest.fixture
    def mock_play_logic(self):
        """Mock do play logic"""
        play_logic = Mock()
        play_logic.attack_distance = 200
        play_logic.shot_cooldown = 0.45
        return play_logic
    
    @pytest.fixture
    def mock_safety_system(self):
        """Mock do safety system"""
        safety = Mock()
        safety.suspicion_threshold = 0.5
        return safety
    
    def test_auto_tuner_initialization(self, auto_tuner):
        """Testa inicialização do auto-tuner"""
        assert auto_tuner.match_controller is not None
        assert auto_tuner.config is not None
        assert auto_tuner.last_tuning_time == 0
        assert auto_tuner.tuning_history == []
    
    def test_should_tune_with_sufficient_matches(self, auto_tuner):
        """Testa se deve fazer tuning com partidas suficientes"""
        assert auto_tuner.should_tune() == True
    
    def test_should_not_tune_with_insufficient_matches(self, mock_match_controller):
        """Testa se não deve fazer tuning com partidas insuficientes"""
        mock_match_controller.get_stats = Mock(return_value={
            "total": 3,
            "wins": 1,
            "losses": 2,
            "draws": 0,
            "win_rate": 33.0,
            "total_trophies": 10,
            "avg_duration": 100.0,
            "total_kills": 5,
            "total_damage": 5000
        })
        
        config = TuningConfig(min_matches_for_tuning=10)
        tuner = AutoTuner(mock_match_controller, config)
        
        assert tuner.should_tune() == False
    
    def test_analyze_performance(self, auto_tuner):
        """Testa análise de performance"""
        analysis = auto_tuner.analyze_performance()
        
        assert "win_rate" in analysis
        assert "avg_kills_per_match" in analysis
        assert "avg_damage_per_match" in analysis
        assert "performance_rating" in analysis
        assert 0 <= analysis["performance_rating"] <= 1
    
    def test_calculate_performance_rating(self, auto_tuner):
        """Testa cálculo de rating de performance"""
        stats = {
            "total": 20,
            "wins": 12,
            "losses": 6,
            "draws": 2,
            "win_rate": 60.0,
            "total_trophies": 200,
            "avg_duration": 120.0,
            "total_kills": 40,
            "total_damage": 20000
        }
        
        rating = auto_tuner._calculate_performance_rating(stats)
        
        assert 0 <= rating <= 1
        assert rating >= 0.5  # Win rate de 60% deve dar rating >= 0.5
    
    def test_calculate_adjustments_low_performance(self, auto_tuner):
        """Testa cálculo de ajustes para performance baixa"""
        analysis = {
            "win_rate": 0.3,
            "avg_kills_per_match": 0.5,
            "avg_damage_per_match": 500,
            "total_matches": 20,
            "performance_rating": 0.3
        }
        
        adjustments = auto_tuner.calculate_adjustments(analysis)
        
        assert adjustments is not None
        # Performance baixa deve aumentar distância de ataque
        assert "attack_distance" in adjustments
        assert adjustments["attack_distance"] > 0
    
    def test_calculate_adjustments_high_performance(self, auto_tuner):
        """Testa cálculo de ajustes para performance alta"""
        analysis = {
            "win_rate": 0.8,
            "avg_kills_per_match": 4.0,
            "avg_damage_per_match": 3000,
            "total_matches": 20,
            "performance_rating": 0.8
        }
        
        adjustments = auto_tuner.calculate_adjustments(analysis)
        
        assert adjustments is not None
        # Performance alta deve diminuir distância (mais agressivo)
        assert "attack_distance" in adjustments
        assert adjustments["attack_distance"] < 0
    
    def test_apply_adjustments(self, auto_tuner, mock_play_logic, mock_safety_system):
        """Testa aplicação de ajustes"""
        adjustments = {
            "attack_distance": 10,
            "shot_cooldown": -5,
            "safety_threshold": 5,
            "aggressiveness": 10
        }
        
        success = auto_tuner.apply_adjustments(adjustments, mock_play_logic, mock_safety_system)
        
        assert success == True
        assert mock_play_logic.attack_distance != 200  # Deve ter mudado
        assert mock_play_logic.shot_cooldown != 0.45  # Deve ter mudado
        assert mock_safety_system.suspicion_threshold != 0.5  # Deve ter mudado
        assert len(auto_tuner.tuning_history) == 1
    
    def test_apply_adjustments_respects_limits(self, auto_tuner, mock_play_logic, mock_safety_system):
        """Testa se ajustes respeitam limites"""
        # Ajuste muito grande
        adjustments = {
            "attack_distance": 1000,  # Deve ser limitado a max_parameter_change_percent
        }
        
        old_distance = mock_play_logic.attack_distance
        success = auto_tuner.apply_adjustments(adjustments, mock_play_logic, mock_safety_system)
        
        assert success == True
        # Deve estar dentro dos limites
        assert auto_tuner.config.min_attack_distance <= mock_play_logic.attack_distance <= auto_tuner.config.max_attack_distance
    
    def test_tune_cycle(self, auto_tuner, mock_play_logic, mock_safety_system):
        """Testa ciclo completo de tuning"""
        # Simular performance baixa
        auto_tuner.match_controller.get_stats = Mock(return_value={
            "total": 20,
            "wins": 5,
            "losses": 15,
            "draws": 0,
            "win_rate": 25.0,
            "total_trophies": -50,
            "avg_duration": 100.0,
            "total_kills": 10,
            "total_damage": 5000
        })
        
        result = auto_tuner.tune(mock_play_logic, mock_safety_system)
        
        assert result["success"] == True
        assert "analysis" in result
        assert "adjustments" in result
        assert "current_params" in result
    
    def test_tune_cycle_no_adjustments_needed(self, auto_tuner, mock_play_logic, mock_safety_system):
        """Testa ciclo de tuning quando não há ajustes necessários"""
        # Simular performance adequada
        auto_tuner.match_controller.get_stats = Mock(return_value={
            "total": 20,
            "wins": 11,
            "losses": 8,
            "draws": 1,
            "win_rate": 55.0,
            "total_trophies": 100,
            "avg_duration": 120.0,
            "total_kills": 25,
            "total_damage": 10000
        })
        
        result = auto_tuner.tune(mock_play_logic, mock_safety_system)
        
        assert result["success"] == False
        assert result["reason"] == "Nenhum ajuste necessário"
    
    def test_get_tuning_status(self, auto_tuner):
        """Testa obtenção de status do tuning"""
        status = auto_tuner.get_tuning_status()
        
        assert "last_tuning_time" in status
        assert "last_tuning_hours_ago" in status
        assert "tuning_history_count" in status
        assert "current_params" in status
        assert "config" in status
    
    def test_reset_params(self, auto_tuner, mock_play_logic, mock_safety_system):
        """Testa reset de parâmetros"""
        # Aplicar alguns ajustes primeiro
        adjustments = {"attack_distance": 10}
        auto_tuner.apply_adjustments(adjustments, mock_play_logic, mock_safety_system)
        
        # Resetar
        success = auto_tuner.reset_params(mock_play_logic, mock_safety_system)
        
        assert success == True
        assert auto_tuner.current_params["attack_distance"] == 200  # Valor padrão
        assert auto_tuner.current_params["aggressiveness"] == 0.5  # Valor padrão
        assert len(auto_tuner.tuning_history) == 0
    
    def test_tuning_interval_respected(self, auto_tuner):
        """Testa se intervalo de tuning é respeitado"""
        config = TuningConfig(
            min_matches_for_tuning=5,
            tuning_interval_hours=1,  # 1 hora
            win_rate_target=0.6
        )
        tuner = AutoTuner(auto_tuner.match_controller, config)
        
        # Primeira verificação deve passar
        assert tuner.should_tune() == True
        
        # Simular tuning recente
        tuner.last_tuning_time = time.time()
        
        # Segunda verificação deve falhar (intervalo não passou)
        assert tuner.should_tune() == False
