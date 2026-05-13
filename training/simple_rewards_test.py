"""
Simple Rewards Test
Teste simples e direto do sistema de rewards
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from training.real_reward_system import RealRewardCalculator, GameMetrics
    
    print("Testing Rewards System...")
    calculator = RealRewardCalculator()
    
    # Teste simples
    metrics = GameMetrics(
        match_id="test_1",
        timestamp=datetime.now().isoformat(),
        kills=5,
        deaths=2,
        damage_dealt=3000,
        damage_taken=1500,
        survival_time=120,
        final_position=3,
        power_cubes_collected=10,
        enemies_detected=15,
        detection_accuracy=0.85,
        good_decisions=30,
        bad_decisions=5,
        decision_accuracy=0.86
    )
    
    reward = calculator.calculate_total_reward(metrics)
    
    print(f"Total Reward: {reward.total_reward}")
    print(f"Normalized Reward: {reward.normalized_reward}")
    print(f"Kill Reward: {reward.kill_reward}")
    print(f"Survival Reward: {reward.survival_reward}")
    print(f"Damage Reward: {reward.damage_reward}")
    print(f"Decision Reward: {reward.decision_reward}")
    print(f"Death Penalty: {reward.death_penalty}")
    
    print("OK Rewards system working!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)