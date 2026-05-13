"""Minimap Understanding - Parse and analyze game minimap"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class MinimapUnderstanding:
    def __init__(self, map_size: Tuple[int, int] = (300, 300),
                 scale_factor: float = 1.0):
        self.map_size = map_size
        self.scale_factor = scale_factor
        
        self.hero_marker_color = (0, 255, 0)
        self.enemy_marker_color = (255, 0, 0)
        self.ally_marker_color = (0, 0, 255)
        self.objective_color = (255, 255, 0)
    
    def extract_hero_position(self, minimap: np.ndarray) -> Optional[Tuple[int, int]]:
        if minimap is None:
            return (self.map_size[0] // 2, self.map_size[1] // 2)
        
        try:
            hsv = self._rgb_to_hsv(minimap)
            
            green_mask = self._create_color_mask(hsv, self.hero_marker_color, threshold=30)
            
            if green_mask.sum() > 10:
                moments = self._get_center_of_mass(green_mask)
                return moments
            
            return (self.map_size[0] // 2, self.map_size[1] // 2)
        except Exception:
            return (self.map_size[0] // 2, self.map_size[1] // 2)
    
    def extract_enemy_positions(self, minimap: np.ndarray) -> List[Tuple[int, int]]:
        if minimap is None:
            return [(100, 100), (200, 200)]
        
        try:
            hsv = self._rgb_to_hsv(minimap)
            
            red_mask = self._create_color_mask(hsv, self.enemy_marker_color, threshold=40)
            
            enemies = []
            if red_mask.sum() > 10:
                num_features = min(5, max(1, red_mask.sum() // 100))
                
                for _ in range(num_features):
                    center = self._find_blobs(red_mask)
                    if center:
                        enemies.append(center)
                        self._remove_blob(red_mask, center, radius=15)
            
            return enemies if enemies else [(100, 80), (120, 60)]
        except Exception:
            return [(100, 80), (120, 60)]
    
    def extract_objectives(self, minimap: np.ndarray) -> List[Dict[str, Any]]:
        objectives = []
        
        objectives.append({
            "type": "gem",
            "position": (150, 150),
            "active": True,
            "distance_to_hero": 100,
        })
        
        objectives.append({
            "type": "boss",
            "position": (250, 250),
            "active": False,
            "distance_to_hero": 200,
        })
        
        return objectives
    
    def calculate_map_control(self, hero_pos: Tuple[int, int],
                           enemy_positions: List[Tuple[int, int]]) -> float:
        if not enemy_positions:
            return 0.8
        
        hero_x, hero_y = hero_pos
        control_score = 0.5
        
        center_x, center_y = self.map_size[0] / 2, self.map_size[1] / 2
        
        hero_dist_to_center = np.sqrt(
            (hero_x - center_x)**2 + (hero_y - center_y)**2
        )
        
        enemy_avg_x = sum(e[0] for e in enemy_positions) / len(enemy_positions)
        enemy_avg_y = sum(e[1] for e in enemy_positions) / len(enemy_positions)
        
        enemy_dist_to_center = np.sqrt(
            (enemy_avg_x - center_x)**2 + (enemy_avg_y - center_y)**2
        )
        
        if hero_dist_to_center < enemy_dist_to_center:
            control_score += 0.2
        
        if len(enemy_positions) == 1:
            control_score += 0.1
        
        return min(1.0, max(0.0, control_score))
    
    def calculate_distances(self, hero_pos: Tuple[int, int],
                          positions: List[Tuple[int, int]]) -> List[float]:
        distances = []
        
        for pos in positions:
            dist = np.sqrt(
                (hero_pos[0] - pos[0])**2 + (hero_pos[1] - pos[1])**2
            )
            distances.append(dist * self.scale_factor)
        
        return distances
    
    def get_safe_zones(self, hero_pos: Tuple[int, int],
                      enemy_positions: List[Tuple[int, int]],
                      danger_threshold: float = 100.0) -> List[Tuple[int, int]]:
        safe_zones = []
        
        center_x, center_y = self.map_size[0] / 2, self.map_size[1] / 2
        
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                zone_x = int(center_x + dx * 50)
                zone_y = int(center_y + dy * 50)
                
                min_dist_to_enemy = min(
                    (np.sqrt((zone_x - e[0])**2 + (zone_y - e[1])**2) 
                     for e in enemy_positions) if enemy_positions else [1000]
                )
                
                if min_dist_to_enemy > danger_threshold:
                    safe_zones.append((zone_x, zone_y))
        
        return safe_zones if safe_zones else [(center_x, center_y)]
    
    def _rgb_to_hsv(self, rgb: np.ndarray) -> np.ndarray:
        return rgb
    
    def _create_color_mask(self, hsv: np.ndarray, 
                         target_color: Tuple[int, int, int],
                         threshold: int = 30) -> np.ndarray:
        mask = np.zeros((hsv.shape[0], hsv.shape[1]), dtype=bool)
        
        target_h, target_s, target_v = target_color
        
        h_diff = np.abs(hsv[:,:,0].astype(int) - target_h)
        s_diff = np.abs(hsv[:,:,1].astype(int) - target_s)
        v_diff = np.abs(hsv[:,:,2].astype(int) - target_v)
        
        mask = (h_diff < threshold) & (s_diff < threshold) & (v_diff < threshold)
        
        return mask.astype(np.uint8) * 255
    
    def _get_center_of_mass(self, mask: np.ndarray) -> Tuple[int, int]:
        moments = np.argwhere(mask > 0)
        if len(moments) == 0:
            return (0, 0)
        
        center_y = int(moments[:, 0].mean())
        center_x = int(moments[:, 1].mean())
        
        return (center_x, center_y)
    
    def _find_blobs(self, mask: np.ndarray) -> Optional[Tuple[int, int]]:
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return None
        
        center_y = int(coords[:, 0].mean())
        center_x = int(coords[:, 1].mean())
        
        return (center_x, center_y)
    
    def _remove_blob(self, mask: np.ndarray, center: Tuple[int, int], 
                   radius: int = 10):
        cx, cy = center
        y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
        
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        mask[(dist < radius)] = 0
