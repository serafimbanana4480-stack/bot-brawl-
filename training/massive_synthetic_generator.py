"""
massive_synthetic_generator.py

Gerador massivo de dados sintéticos para treinamento.
Gera apenas dados estruturados (estado-ação) sem imagens para velocidade.
Cria 10.000+ amostras rapidamente para BC e CQL.

Funcionalidades:
- Geração massiva de estados de jogo
- Trajetórias realistas com física
- Diversidade de cenários de combate
- Data augmentation de estados
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GameState:
    """Estado do jogo para BC/CQL"""
    player_x: float
    player_y: float
    health: float
    ammo: int
    enemy_distance: float
    enemy_angle: float
    obstacle_nearby: int
    powerup_available: int


@dataclass
class Action:
    """Ação do jogador"""
    move_direction: str  # "up", "down", "left", "right", "none"
    attack: int
    use_ability: int
    target_x: float
    target_y: float


@dataclass
class Transition:
    """Transição estado-ação-estado para CQL"""
    state: List[float]
    action: List[float]
    reward: float
    next_state: List[float]
    done: bool


class MassiveSyntheticGenerator:
    """Gerador massivo de dados sintéticos"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_game_state(self) -> GameState:
        """Gera um estado de jogo aleatório"""
        return GameState(
            player_x=random.random() * 1920,
            player_y=random.random() * 1080,
            health=random.uniform(0, 100),
            ammo=random.randint(0, 10),
            enemy_distance=random.uniform(50, 500),
            enemy_angle=random.uniform(0, 360),
            obstacle_nearby=random.randint(0, 1),
            powerup_available=random.randint(0, 1)
        )
    
    def generate_action(self, state: GameState) -> Action:
        """Gera uma ação baseada no estado (policy simulada)"""
        # Política simples: mover em direção ao inimigo se longe, atacar se perto
        if state.enemy_distance > 200:
            # Mover em direção ao inimigo
            directions = ["up", "down", "left", "right"]
            move_direction = random.choice(directions)
            attack = 0
        else:
            # Atacar
            move_direction = "none"
            attack = 1 if state.ammo > 0 else 0
        
        return Action(
            move_direction=move_direction,
            attack=attack,
            use_ability=random.randint(0, 1),
            target_x=random.random() * 1920,
            target_y=random.random() * 1080
        )
    
    def generate_bc_dataset(self, num_samples: int = 10000) -> Dict:
        """Gera dataset de Behavior Cloning massivo"""
        logger.info(f"Gerando dataset BC com {num_samples} amostras...")
        
        episodes = []
        episode_size = 50  # frames por episode
        
        for episode_id in range(num_samples // episode_size):
            frames = []
            
            # Posição inicial
            player_x, player_y = random.random() * 1920, random.random() * 1080
            
            for frame_id in range(episode_size):
                # Simular movimento
                player_x += random.uniform(-50, 50)
                player_y += random.uniform(-50, 50)
                player_x = max(0, min(1920, player_x))
                player_y = max(0, min(1080, player_y))
                
                state = GameState(
                    player_x=player_x,
                    player_y=player_y,
                    health=random.uniform(0, 100),
                    ammo=random.randint(0, 10),
                    enemy_distance=random.uniform(50, 500),
                    enemy_angle=random.uniform(0, 360),
                    obstacle_nearby=random.randint(0, 1),
                    powerup_available=random.randint(0, 1)
                )
                
                action = self.generate_action(state)
                
                frames.append({
                    "frame_id": frame_id,
                    "state": asdict(state),
                    "action": asdict(action)
                })
            
            episodes.append({
                "episode_id": episode_id,
                "length": episode_size,
                "frames": frames
            })
            
            if episode_id % 100 == 0:
                logger.info(f"Progresso: {episode_id * episode_size}/{num_samples} amostras")
        
        # Salvar dataset
        bc_path = self.output_dir / "bc_massive.json"
        with open(bc_path, 'w') as f:
            json.dump(episodes, f, indent=2)
        
        logger.info(f"Dataset BC salvo: {bc_path}")
        return {"num_episodes": len(episodes), "total_frames": len(episodes) * episode_size}
    
    def generate_cql_dataset(self, num_transitions: int = 50000) -> Dict:
        """Gera replay buffer massivo para CQL"""
        logger.info(f"Gerando replay buffer CQL com {num_transitions} transições...")
        
        transitions = []
        
        for i in range(num_transitions):
            # Estado atual
            state = [
                random.random(),  # player_x norm
                random.random(),  # player_y norm
                random.random(),  # health norm
                random.random(),  # ammo norm
                random.random(),  # enemy_distance norm
                random.random(),  # enemy_angle norm
                random.randint(0, 1),  # obstacle_nearby
                random.randint(0, 1)  # powerup_available
            ]
            
            # Ação (5 ações possíveis: 4 direções + none)
            action = [0.0] * 5
            action_idx = random.randint(0, 4)
            action[action_idx] = 1.0
            
            # Reward (simulado)
            reward = random.uniform(-1, 1)
            
            # Próximo estado
            next_state = [
                min(1.0, max(0.0, state[0] + random.uniform(-0.1, 0.1))),
                min(1.0, max(0.0, state[1] + random.uniform(-0.1, 0.1))),
                min(1.0, max(0.0, state[2] + random.uniform(-0.05, 0.05))),
                random.random(),
                min(1.0, max(0.0, state[4] + random.uniform(-0.1, 0.1))),
                random.random(),
                random.randint(0, 1),
                random.randint(0, 1)
            ]
            
            # Done (episode termination)
            done = random.random() < 0.01  # 1% chance de terminar
            
            transitions.append({
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": done
            })
            
            if i % 5000 == 0:
                logger.info(f"Progresso: {i}/{num_transitions} transições")
        
        # Salvar replay buffer
        cql_path = self.output_dir / "replay_buffer_massive.json"
        with open(cql_path, 'w') as f:
            json.dump(transitions, f, indent=2)
        
        logger.info(f"Replay buffer CQL salvo: {cql_path}")
        return {"num_transitions": len(transitions)}
    
    def generate_yolo_labels(self, num_samples: int = 5000) -> Dict:
        """Gera labels YOLO massivos"""
        logger.info(f"Gerando labels YOLO com {num_samples} amostras...")
        
        labels_dir = self.output_dir / "yolo_labels"
        labels_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            # Gerar labels aleatórios
            labels = []
            
            # Player (class 0)
            px, py = random.random(), random.random()
            labels.append(f"0 {px:.6f} {py:.6f} 0.05 0.05")
            
            # Enemies (class 1) - 0-3 enemies
            num_enemies = random.randint(0, 3)
            for _ in range(num_enemies):
                ex, ey = random.random(), random.random()
                labels.append(f"1 {ex:.6f} {ey:.6f} 0.05 0.05")
            
            # Bushes (class 2) - 0-5 bushes
            num_bushes = random.randint(0, 5)
            for _ in range(num_bushes):
                bx, by = random.random(), random.random()
                bw, bh = random.uniform(0.05, 0.15), random.uniform(0.05, 0.15)
                labels.append(f"2 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
            
            # Power cubes (class 3) - 0-2 cubes
            num_cubes = random.randint(0, 2)
            for _ in range(num_cubes):
                cx, cy = random.random(), random.random()
                labels.append(f"3 {cx:.6f} {cy:.6f} 0.03 0.03")
            
            # Salvar label file
            label_path = labels_dir / f"sample_{i:06d}.txt"
            with open(label_path, 'w') as f:
                f.write('\n'.join(labels))
            
            if i % 500 == 0:
                logger.info(f"Progresso: {i}/{num_samples} labels")
        
        logger.info(f"Labels YOLO salvos: {labels_dir}")
        return {"num_labels": num_samples}


def main():
    """Função principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Massive Synthetic Data Generator")
    parser.add_argument("--bc-samples", type=int, default=10000, help="Amostras BC")
    parser.add_argument("--cql-transitions", type=int, default=50000, help="Transições CQL")
    parser.add_argument("--yolo-samples", type=int, default=5000, help="Labels YOLO")
    parser.add_argument("--output", default="./dataset/synthetic_massive", help="Output dir")
    
    args = parser.parse_args()
    
    # Criar gerador
    generator = MassiveSyntheticGenerator(Path(args.output))
    
    # Gerar datasets
    stats = {}
    
    if args.bc_samples > 0:
        stats["bc"] = generator.generate_bc_dataset(args.bc_samples)
    
    if args.cql_transitions > 0:
        stats["cql"] = generator.generate_cql_dataset(args.cql_transitions)
    
    if args.yolo_samples > 0:
        stats["yolo"] = generator.generate_yolo_labels(args.yolo_samples)
    
    logger.info("=" * 60)
    logger.info("GERAÇÃO MASSIVA COMPLETA")
    logger.info("=" * 60)
    logger.info(f"Output: {args.output}")
    logger.info(f"Estatísticas: {stats}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()