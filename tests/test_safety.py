import unittest
import time
import sys
from pathlib import Path


from backend.brawl_bot.safety_system import SafetySystem, SafetyConfig

class TestSafetySystem(unittest.TestCase):
    def setUp(self):
        self.config = SafetyConfig(
            max_trophies=100,
            min_apm=10,
            max_apm=50,
            max_session_hours=0.01 # ~36 segundos para teste rápido
        )
        self.safety = SafetySystem(self.config)

    def test_apm_monitoring(self):
        """Testa se o sistema detecta excesso de cliques (Anti-Ban)"""
        self.safety.start_session()
        
        # Simular 100 ações num curto espaço de tempo
        for _ in range(100):
            self.safety.record_action()
            
        status = self.safety.get_status()
        self.assertGreater(status['current_apm'], 50)
        print(f"DEBUG: APM Detectado: {status['current_apm']}")

    def test_trophy_limit(self):
        """Testa se o bot para ao atingir a meta"""
        status = self.safety.check_trophy_limit(110)
        self.assertFalse(status['can_play'])
        self.assertIn("Meta de troféus", status['message'])

if __name__ == '__main__':
    unittest.main()
