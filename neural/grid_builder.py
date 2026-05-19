"""
Spatial Grid Builder for Neural Network Input

Converts YOLO detections into a 21x21 spatial grid centered on the player.
This grid serves as input to the SpatialCNN for spatial feature extraction.

Grid Specification:
- Size: 21x21 cells
- Cell size: 50 pixels (covers 1050px range in all directions)
- Channels: 35 (one per visual class in full schema)
- Encoding: One-hot per cell (0 or 1)

Usage:
    from neural.grid_builder import SpatialGridBuilder
    from core.class_registry import get_schema
    
    builder = SpatialGridBuilder()
    grid = builder.build(player_pos, detections, schema="full")
    # grid.shape = (21, 21, 35)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class SpatialGridBuilder:
    """
    Builds 21x21 spatial grid centered on player for CNN input.
    
    The grid represents the local environment around the player,
    with each cell indicating the presence of game objects.
    """
    
    GRID_SIZE = 21
    CELL_SIZE_PX = 50  # Each cell represents 50x50 pixels
    GRID_RANGE_PX = GRID_SIZE * CELL_SIZE_PX  # 1050px total range
    
    def __init__(self, grid_size: int = 21, cell_size: int = 50):
        """
        Initialize grid builder.
        
        Args:
            grid_size: Grid dimension (N x N)
            cell_size: Pixel size of each grid cell
        """
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.grid_range = grid_size * cell_size
        
    def build(
        self,
        player_pos: Tuple[float, float],
        detections: Dict[str, List],
        schema: str = "full"
    ) -> np.ndarray:
        """
        Build spatial grid from detections.
        
        Args:
            player_pos: Player (x, y) position as center
            detections: Detection dict {class_name: [bbox, ...]}
            schema: Schema tier ("core", "extended", "full")
            
        Returns:
            Grid array of shape (GRID_SIZE, GRID_SIZE, num_classes) with one-hot encoding
        """
        from core.class_registry import get_schema, get_class_id
        
        # Get schema to determine number of classes
        schema_dict = get_schema(schema)
        num_classes = len(schema_dict)
        
        # Initialize empty grid
        grid = np.zeros((self.grid_size, self.grid_size, num_classes), dtype=np.float32)
        
        # Center of grid in world coordinates
        px, py = player_pos
        
        # Process each detection
        for class_name, boxes in detections.items():
            class_id = get_class_id(class_name, schema=schema)
            
            if class_id is None or class_id >= num_classes:
                continue  # Skip classes not in target schema
            
            for bbox in boxes:
                # Get center of bounding box
                if len(bbox) >= 4:
                    bx = (bbox[0] + bbox[2]) / 2
                    by = (bbox[1] + bbox[3]) / 2
                else:
                    bx, by = bbox[0], bbox[1]
                
                # Convert to grid coordinates relative to player
                gx, gy = self._world_to_grid(px, py, bx, by)
                
                # Check if within grid bounds
                if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                    # One-hot encode (set to 1.0)
                    grid[gy, gx, class_id] = 1.0
        
        return grid
    
    def _world_to_grid(
        self,
        player_x: float,
        player_y: float,
        world_x: float,
        world_y: float
    ) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates.
        
        Args:
            player_x, player_y: Player position (center)
            world_x, world_y: Object position in world coordinates
            
        Returns:
            (grid_x, grid_y) tuple
        """
        # Calculate relative position
        rel_x = world_x - player_x
        rel_y = world_y - player_y
        
        # Convert to grid cell coordinates
        # Add half grid range to shift to positive coordinates
        grid_x = int((rel_x + self.grid_range / 2) / self.cell_size)
        grid_y = int((rel_y + self.grid_range / 2) / self.cell_size)
        
        return grid_x, grid_y
    
    def grid_to_world(
        self,
        player_x: float,
        player_y: float,
        grid_x: int,
        grid_y: int
    ) -> Tuple[float, float]:
        """
        Convert grid coordinates back to world coordinates.
        
        Args:
            player_x, player_y: Player position (center)
            grid_x, grid_y: Grid cell coordinates
            
        Returns:
            (world_x, world_y) tuple representing center of grid cell
        """
        # Convert to relative coordinates
        rel_x = (grid_x * self.cell_size) + (self.cell_size / 2) - (self.grid_range / 2)
        rel_y = (grid_y * self.cell_size) + (self.cell_size / 2) - (self.grid_range / 2)
        
        # Add player position
        world_x = player_x + rel_x
        world_y = player_y + rel_y
        
        return world_x, world_y
    
    def get_grid_info(self) -> Dict:
        """
        Get grid configuration information.
        
        Returns:
            Dict with grid parameters
        """
        return {
            "grid_size": self.grid_size,
            "cell_size": self.cell_size,
            "grid_range_px": self.grid_range,
            "total_cells": self.grid_size * self.grid_size,
        }
