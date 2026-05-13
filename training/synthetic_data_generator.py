"""
synthetic_data_generator.py

Gerador de dados sintéticos realistas para treinamento de Brawl Stars bot.
Cria dados que simulam gameplay real com física básica e padrões realistas.

Funcionalidades:
- Geração de imagens sintéticas com templates reais do jogo
- Simulação de física básica (movimento, projéteis)
- Criação de trajetórias realistas para personagens
- Geração de estados de jogo variados (lobby, combate, etc.)
- Data augmentation avançada (perspectiva, iluminação, etc.)
- Metadados enriquecidos para treinamento

Usage:
    python -m brawl_bot.training.synthetic_data_generator --num-samples 1000 --output ./dataset/synthetic
"""

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class SyntheticGameState:
    """Estado do jogo sintético"""
    player_position: Tuple[float, float]  # normalized 0-1
    player_health: float
    player_ammo: int
    enemies: List[Dict]  # enemy positions and states
    projectiles: List[Dict]  # projectile positions and velocities
    bushes: List[Dict]  # bush positions
    power_cubes: List[Dict]  # power cube positions
    game_phase: str  # "lobby", "matchmaking", "game", "result"
    timestamp: float


class PhysicsEngine:
    """Motor de física simples para simulação realista"""
    
    def __init__(self):
        self.gravity = 9.8
        self.friction = 0.95
        self.max_speed = 0.02  # normalized per frame
    
    def apply_movement(self, position: Tuple[float, float], 
                      velocity: Tuple[float, float],
                      dt: float = 1.0) -> Tuple[float, float]:
        """Aplica movimento com física básica"""
        x, y = position
        vx, vy = velocity
        
        # Aplicar velocidade
        new_x = x + vx * dt
        new_y = y + vy * dt
        
        # Aplicar fricção
        vx *= self.friction
        vy *= self.friction
        
        # Limitar posição aos bounds [0, 1]
        new_x = max(0.0, min(1.0, new_x))
        new_y = max(0.0, min(1.0, new_y))
        
        return (new_x, new_y), (vx, vy)
    
    def predict_projectile(self, start_pos: Tuple[float, float],
                          target_pos: Tuple[float, float],
                          speed: float = 0.03) -> Tuple[float, float]:
        """Prediz posição de projétil com física básica"""
        x1, y1 = start_pos
        x2, y2 = target_pos
        
        # Calcular direção
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return start_pos
        
        # Normalizar e aplicar velocidade
        vx = (dx / distance) * speed
        vy = (dy / distance) * speed
        
        return (x1 + vx, y1 + vy)


class TrajectoryGenerator:
    """Gerador de trajetórias realistas para personagens"""
    
    def __init__(self):
        self.current_waypoints = []
        self.current_target = None
    
    def generate_realistic_trajectory(self, start_pos: Tuple[float, float],
                                     duration: int = 30) -> List[Tuple[float, float]]:
        """Gera trajetória realista usando waypoints"""
        trajectory = [start_pos]
        current_pos = start_pos
        
        # Gerar waypoints aleatórios mas coerentes
        num_waypoints = random.randint(3, 8)
        waypoints = []
        
        for _ in range(num_waypoints):
            # Waypoint próximo ao anterior (movimento natural)
            offset_x = random.uniform(-0.15, 0.15)
            offset_y = random.uniform(-0.15, 0.15)
            new_x = max(0.1, min(0.9, current_pos[0] + offset_x))
            new_y = max(0.1, min(0.9, current_pos[1] + offset_y))
            waypoints.append((new_x, new_y))
            current_pos = (new_x, new_y)
        
        # Interpolar entre waypoints
        for i in range(len(waypoints) - 1):
            start = waypoints[i]
            end = waypoints[i + 1]
            segment_length = duration // num_waypoints
            
            for t in range(segment_length):
                alpha = t / segment_length
                # Usar curva de Bézier para movimento suave
                x = start[0] + (end[0] - start[0]) * alpha
                y = start[1] + (end[1] - start[1]) * alpha
                
                # Adicionar pequeno ruído para naturalidade
                noise_x = random.uniform(-0.005, 0.005)
                noise_y = random.uniform(-0.005, 0.005)
                
                trajectory.append((x + noise_x, y + noise_y))
        
        return trajectory
    
    def generate_combat_trajectory(self, player_pos: Tuple[float, float],
                                   enemy_pos: Tuple[float, float],
                                   duration: int = 30) -> List[Tuple[float, float]]:
        """Gera trajetória de combate (perseguição/evasão)"""
        trajectory = []
        current_pos = enemy_pos
        
        for i in range(duration):
            # Decidir entre perseguir ou evadir
            if random.random() < 0.6:
                # Perseguir jogador
                dx = player_pos[0] - current_pos[0]
                dy = player_pos[1] - current_pos[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0:
                    speed = 0.01
                    new_x = current_pos[0] + (dx / distance) * speed
                    new_y = current_pos[1] + (dy / distance) * speed
                    current_pos = (new_x, new_y)
            else:
                # Evadir movimento aleatório
                offset_x = random.uniform(-0.02, 0.02)
                offset_y = random.uniform(-0.02, 0.02)
                new_x = max(0.1, min(0.9, current_pos[0] + offset_x))
                new_y = max(0.1, min(0.9, current_pos[1] + offset_y))
                current_pos = (new_x, new_y)
            
            trajectory.append(current_pos)
        
        return trajectory


class ImageAugmenter:
    """Data augmentation avançado para imagens"""
    
    @staticmethod
    def add_perspective_transform(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
        """Adiciona transformação de perspectiva"""
        h, w = image.shape[:2]
        
        # Pontos de origem
        src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        
        # Pontos de destino com pequena distorção
        offset = strength * min(w, h)
        dst_points = np.float32([
            [random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), random.uniform(-offset, offset)],
            [w + random.uniform(-offset, offset), h + random.uniform(-offset, offset)],
            [random.uniform(-offset, offset), h + random.uniform(-offset, offset)]
        ])
        
        # Calcular matriz de transformação
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Aplicar transformação
        result = cv2.warpPerspective(image, matrix, (w, h))
        
        return result
    
    @staticmethod
    def adjust_lighting(image: np.ndarray, brightness: float = 0.0,
                       contrast: float = 0.0) -> np.ndarray:
        """Ajusta iluminação da imagem"""
        # Brightness
        if brightness != 0:
            image = image + brightness
        
        # Contrast
        if contrast != 0:
            f = 131 * (contrast + 127) / (127 * (131 - contrast))
            alpha_c = f
            gamma_c = 127 * (1 - f)
            
            image = alpha_c * image + gamma_c
            image = np.clip(image, 0, 255).astype(np.uint8)
        
        return image
    
    @staticmethod
    def add_noise(image: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
        """Adiciona ruído gaussiano"""
        noise = np.random.normal(0, noise_level * 255, image.shape)
        image = image + noise
        image = np.clip(image, 0, 255).astype(np.uint8)
        return image
    
    @staticmethod
    def random_augment(image: np.ndarray) -> np.ndarray:
        """Aplica augmentação aleatória"""
        # Perspective transform (30% chance)
        if random.random() < 0.3:
            image = ImageAugmenter.add_perspective_transform(image, strength=0.05)
        
        # Lighting adjustment (50% chance)
        if random.random() < 0.5:
            brightness = random.uniform(-20, 20)
            contrast = random.uniform(-0.2, 0.2)
            image = ImageAugmenter.adjust_lighting(image, brightness, contrast)
        
        # Noise (20% chance)
        if random.random() < 0.2:
            image = ImageAugmenter.add_noise(image, noise_level=0.005)
        
        return image


class SyntheticDataGenerator:
    """Gerador de dados sintéticos realistas"""
    
    def __init__(self, output_dir: Path, use_real_templates: bool = True):
        self.output_dir = Path(output_dir)
        self.use_real_templates = use_real_templates
        
        # Criar diretórios
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.metadata_dir = self.output_dir / "metadata"
        
        for dir_path in [self.images_dir, self.labels_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Componentes
        self.physics = PhysicsEngine()
        self.trajectory_gen = TrajectoryGenerator()
        self.augmenter = ImageAugmenter()
        
        # Templates reais (se disponíveis)
        self.templates = {}
        if use_real_templates:
            self._load_templates()
    
    def _load_templates(self):
        """Carrega templates reais do jogo"""
        template_dir = Path(__file__).parent.parent / "images"
        
        template_files = ["player.png", "enemy.png", "bush.png", "projectile.png"]
        
        for template_file in template_files:
            template_path = template_dir / template_file
            if template_path.exists():
                try:
                    template = cv2.imread(str(template_path), cv2.IMREAD_UNCHANGED)
                    if template is not None:
                        self.templates[template_file.replace(".png", "")] = template
                        logger.info(f"Template carregado: {template_file}")
                except Exception as e:
                    logger.warning(f"Falha ao carregar template {template_file}: {e}")
    
    def generate_synthetic_image(self, game_state: SyntheticGameState) -> np.ndarray:
        """Gera imagem sintética baseada no estado do jogo"""
        # Criar canvas base
        image = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Adicionar background (cor de campo)
        background_color = (34, 139, 34)  # Forest green
        image[:] = background_color
        
        # Adicionar bushes
        for bush in game_state.bushes:
            self._draw_bush(image, bush)
        
        # Adicionar power cubes
        for cube in game_state.power_cubes:
            self._draw_power_cube(image, cube)
        
        # Adicionar player
        self._draw_player(image, game_state.player_position, game_state.player_health)
        
        # Adicionar enemies
        for enemy in game_state.enemies:
            self._draw_enemy(image, enemy)
        
        # Adicionar projéteis
        for proj in game_state.projectiles:
            self._draw_projectile(image, proj)
        
        # Aplicar augmentação
        image = self.augmenter.random_augment(image)
        
        return image
    
    def _draw_bush(self, image: np.ndarray, bush: Dict):
        """Desenha bush na imagem"""
        x, y = bush["position"]
        w, h = bush.get("size", (60, 60))
        
        # Converter coordenadas normalizadas para pixels
        px = int(x * 1920)
        py = int(y * 1080)
        pw = int(w)
        ph = int(h)
        
        # Desenhar bush como ellipse verde
        cv2.ellipse(image, (px, py), (pw//2, ph//2), 0, 0, 360, (0, 100, 0), -1)
        
        # Adicionar textura
        cv2.ellipse(image, (px, py), (pw//3, ph//3), 0, 0, 360, (0, 80, 0), -1)
    
    def _draw_power_cube(self, image: np.ndarray, cube: Dict):
        """Desenha power cube na imagem"""
        x, y = cube["position"]
        size = cube.get("size", 30)
        
        px = int(x * 1920)
        py = int(y * 1080)
        s = int(size)
        
        # Desenhar cubo como quadrado azul
        cv2.rectangle(image, (px - s//2, py - s//2), (px + s//2, py + s//2), 
                    (0, 0, 255), -1)
        cv2.rectangle(image, (px - s//2, py - s//2), (px + s//2, py + s//2), 
                    (255, 255, 255), 2)
    
    def _draw_player(self, image: np.ndarray, position: Tuple[float, float], health: float):
        """Desenha player na imagem"""
        x, y = position
        px = int(x * 1920)
        py = int(y * 1080)
        
        # Desenhar player como círculo azul
        color = (0, 0, 255) if health > 0.5 else (0, 0, 150)
        cv2.circle(image, (px, py), 25, color, -1)
        
        # Adicionar indicador de saúde
        health_bar_width = int(50 * health)
        cv2.rectangle(image, (px - 25, py - 35), (px - 25 + health_bar_width, py - 30),
                     (0, 255, 0), -1)
        cv2.rectangle(image, (px - 25, py - 35), (px + 25, py - 30),
                     (255, 255, 255), 2)
    
    def _draw_enemy(self, image: np.ndarray, enemy: Dict):
        """Desenha enemy na imagem"""
        x, y = enemy["position"]
        health = enemy.get("health", 1.0)
        
        px = int(x * 1920)
        py = int(y * 1080)
        
        # Desenhar enemy como círculo vermelho
        color = (255, 0, 0) if health > 0.5 else (150, 0, 0)
        cv2.circle(image, (px, py), 25, color, -1)
        
        # Adicionar indicador de saúde
        health_bar_width = int(50 * health)
        cv2.rectangle(image, (px - 25, py - 35), (px - 25 + health_bar_width, py - 30),
                     (0, 255, 0), -1)
        cv2.rectangle(image, (px - 25, py - 35), (px + 25, py - 30),
                     (255, 255, 255), 2)
    
    def _draw_projectile(self, image: np.ndarray, proj: Dict):
        """Desenha projétil na imagem"""
        x, y = proj["position"]
        
        px = int(x * 1920)
        py = int(y * 1080)
        
        # Desenhar projétil como pequeno círculo amarelo
        cv2.circle(image, (px, py), 8, (255, 255, 0), -1)
    
    def generate_dataset(self, num_samples: int, sequence_length: int = 10) -> Dict:
        """Gera dataset completo de dados sintéticos"""
        logger.info(f"Gerando {num_samples} amostras sintéticas...")
        
        samples_generated = 0
        stats = {
            "total_samples": 0,
            "by_phase": {},
            "avg_enemies": 0,
            "avg_projectiles": 0
        }
        
        for i in range(num_samples):
            # Escolher fase do jogo aleatoriamente
            phases = ["lobby", "matchmaking", "game", "result"]
            phase_weights = [0.2, 0.1, 0.6, 0.1]  # Mais gameplay
            game_phase = random.choices(phases, weights=phase_weights)[0]
            
            # Gerar estado do jogo
            game_state = self._generate_random_game_state(game_phase)
            
            # Gerar sequência de frames
            for frame_idx in range(sequence_length):
                # Atualizar estado com física
                game_state = self._update_game_state_physics(game_state)
                
                # Gerar imagem
                image = self.generate_synthetic_image(game_state)
                
                # Salvar imagem
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_")
                stem = f"{timestamp}{i:06d}_{frame_idx:03d}"
                
                image_path = self.images_dir / f"{stem}.png"
                label_path = self.labels_dir / f"{stem}.txt"
                meta_path = self.metadata_dir / f"{stem}.json"
                
                cv2.imwrite(str(image_path), image)
                
                # Gerar labels YOLO
                self._generate_yolo_labels(label_path, game_state, image.shape[:2])
                
                # Salvar metadados
                metadata = {
                    "timestamp": datetime.now().isoformat(),
                    "game_phase": game_phase,
                    "frame_index": frame_idx,
                    "player_position": game_state.player_position,
                    "player_health": game_state.player_health,
                    "player_ammo": game_state.player_ammo,
                    "enemy_count": len(game_state.enemies),
                    "projectile_count": len(game_state.projectiles),
                    "bush_count": len(game_state.bushes)
                }
                
                with open(meta_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                samples_generated += 1
                
                # Atualizar estatísticas
                stats["total_samples"] += 1
                stats["by_phase"][game_phase] = stats["by_phase"].get(game_phase, 0) + 1
                stats["avg_enemies"] = (stats["avg_enemies"] * (samples_generated - 1) + 
                                       len(game_state.enemies)) / samples_generated
                stats["avg_projectiles"] = (stats["avg_projectiles"] * (samples_generated - 1) + 
                                           len(game_state.projectiles)) / samples_generated
                
                if samples_generated % 100 == 0:
                    logger.info(f"Geradas {samples_generated} amostras...")
        
        # Salvar estatísticas
        stats_path = self.output_dir / "generation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset sintético gerado: {samples_generated} amostras")
        logger.info(f"Estatísticas: {stats}")
        
        return stats
    
    def _generate_random_game_state(self, game_phase: str) -> SyntheticGameState:
        """Gera estado do jogo aleatório"""
        # Posição do player
        player_pos = (random.uniform(0.2, 0.8), random.uniform(0.2, 0.8))
        player_health = random.uniform(0.3, 1.0)
        player_ammo = random.randint(0, 3)
        
        # Enemies (mais em gameplay)
        num_enemies = 0 if game_phase == "lobby" else random.randint(1, 5)
        enemies = []
        for _ in range(num_enemies):
            enemy_pos = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            enemies.append({
                "position": enemy_pos,
                "health": random.uniform(0.5, 1.0),
                "velocity": (random.uniform(-0.01, 0.01), random.uniform(-0.01, 0.01))
            })
        
        # Projectiles
        num_projectiles = 0 if game_phase == "lobby" else random.randint(0, 10)
        projectiles = []
        for _ in range(num_projectiles):
            proj_pos = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            projectiles.append({
                "position": proj_pos,
                "velocity": (random.uniform(-0.02, 0.02), random.uniform(-0.02, 0.02))
            })
        
        # Bushes
        num_bushes = random.randint(3, 10)
        bushes = []
        for _ in range(num_bushes):
            bush_pos = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            bushes.append({
                "position": bush_pos,
                "size": (random.randint(40, 80), random.randint(40, 80))
            })
        
        # Power cubes
        num_cubes = 0 if game_phase == "lobby" else random.randint(0, 5)
        power_cubes = []
        for _ in range(num_cubes):
            cube_pos = (random.uniform(0.1, 0.9), random.uniform(0.1, 0.9))
            power_cubes.append({
                "position": cube_pos,
                "size": random.randint(20, 40)
            })
        
        return SyntheticGameState(
            player_position=player_pos,
            player_health=player_health,
            player_ammo=player_ammo,
            enemies=enemies,
            projectiles=projectiles,
            bushes=bushes,
            power_cubes=power_cubes,
            game_phase=game_phase,
            timestamp=time.time()
        )
    
    def _update_game_state_physics(self, game_state: SyntheticGameState) -> SyntheticGameState:
        """Atualiza estado do jogo com física"""
        # Atualizar player position (movimento aleatório)
        player_vel = (random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005))
        new_player_pos, _ = self.physics.apply_movement(game_state.player_position, player_vel)
        
        # Atualizar enemies
        updated_enemies = []
        for enemy in game_state.enemies:
            new_pos, new_vel = self.physics.apply_movement(
                enemy["position"], enemy["velocity"]
            )
            updated_enemies.append({
                "position": new_pos,
                "health": enemy["health"],
                "velocity": new_vel
            })
        
        # Atualizar projectiles
        updated_projectiles = []
        for proj in game_state.projectiles:
            new_pos, _ = self.physics.apply_movement(
                proj["position"], proj["velocity"]
            )
            updated_projectiles.append({
                "position": new_pos,
                "velocity": proj["velocity"]
            })
        
        return SyntheticGameState(
            player_position=new_player_pos,
            player_health=game_state.player_health,
            player_ammo=game_state.player_ammo,
            enemies=updated_enemies,
            projectiles=updated_projectiles,
            bushes=game_state.bushes,
            power_cubes=game_state.power_cubes,
            game_phase=game_state.game_phase,
            timestamp=time.time()
        )
    
    def _generate_yolo_labels(self, label_path: Path, game_state: SyntheticGameState, 
                            image_shape: Tuple[int, int]):
        """Gera labels no formato YOLO"""
        h, w = image_shape
        
        labels = []
        
        # Player (class 0)
        px, py = game_state.player_position
        labels.append(f"0 {px:.6f} {py:.6f} 0.05 0.05")
        
        # Enemies (class 1)
        for enemy in game_state.enemies:
            ex, ey = enemy["position"]
            labels.append(f"1 {ex:.6f} {ey:.6f} 0.05 0.05")
        
        # Bushes (class 2)
        for bush in game_state.bushes:
            bx, by = bush["position"]
            size = bush["size"]
            bw, bh = size[0] / w, size[1] / h
            labels.append(f"2 {bx:.6f} {by:.6f} {bw:.6f} {bh:.6f}")
        
        # Power cubes (class 3)
        for cube in game_state.power_cubes:
            cx, cy = cube["position"]
            size = cube["size"]
            cw, ch = size / w, size / h
            labels.append(f"3 {cx:.6f} {cy:.6f} {cw:.6f} {ch:.6f}")
        
        # Projectiles (class 4)
        for proj in game_state.projectiles:
            prx, pry = proj["position"]
            labels.append(f"4 {prx:.6f} {pry:.6f} 0.01 0.01")
        
        with open(label_path, 'w') as f:
            f.write('\n'.join(labels))


import time

def main():
    """Função principal para execução via linha de comando"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Synthetic Data Generator")
    parser.add_argument("--num-samples", type=int, default=1000, help="Número de amostras")
    parser.add_argument("--sequence-length", type=int, default=10, help="Comprimento da sequência")
    parser.add_argument("--output", default="./dataset/synthetic", help="Diretório de output")
    parser.add_argument("--no-templates", action="store_true", help="Não usar templates reais")
    
    args = parser.parse_args()
    
    # Criar gerador
    generator = SyntheticDataGenerator(
        output_dir=Path(args.output),
        use_real_templates=not args.no_templates
    )
    
    # Gerar dataset
    stats = generator.generate_dataset(args.num_samples, args.sequence_length)
    
    logger.info("Dataset sintético gerado com sucesso!")
    logger.info(f"Output: {args.output}")
    logger.info(f"Estatísticas: {stats}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
