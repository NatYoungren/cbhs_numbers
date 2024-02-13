# For Casco Bay High School
# Nathaniel Alden Homans Youngren
# February 12, 2024

import math
import numpy as np
from bisect import bisect

class Canvas:
    def __init__(self, width: int, height: int, brush_scale: float = None):
        # Canvas properties
        self.width = width          # Width of the canvas in cells.
        self.height = height        # Height of the canvas in cells.
        self.pixels = None          # Array of pixel values (0 or 1).
        self.reset()                # Sets pixels to a blank state.
        
        # Brush properties
        self.prev_brush_pos = None                  # Stored value of the last brush position.
        self.brush_radius = None                    # Radius of the brush in cells.
        self.set_brush_radius(radius=brush_scale)   # Sets the brush radius (or randomizes it).
    

    def reset(self):
        """ Reset the canvas to a blank state.
        """
        self.pixels = np.zeros((self.height, self.width), dtype=np.uint8)
    
    
    def set_brush_radius(self, radius: float = None, rand_range: tuple = (0.35, 0.55)):
        """ Set the radius of the brush, randomly if no radius is given.

        Args:
            radius (float, optional): Override value of the radius. Defaults to None.
            rand_range (tuple, optional): If radius is None, will randomly select from this range. Defaults to (0.25, 0.65).
        """
        if radius is None:
            radius = np.random.uniform(*rand_range)
        
        self.brush_radius = radius


    def draw(self, pos: tuple, color: int, fill_gaps: bool = True):
        """ Colors the cells around the given brush position.
            Colors cells between the previous brush position and the new one if fill_gaps is True.
        
        Args:
            pos (tuple): Position of brush in grid coordinates (float x, float y)
            color (int): Color to fill cells with (0 or 1).
            fill_gaps (bool, optional): If True, will color the space between the previous and new position. Defaults to True.
        """
        # Color the cells around the brush position.
        self.brush_cells(pos, color)
        
        # If fill_gaps is True, color the cells between the new and previous brush positions.
        if fill_gaps and self.prev_brush_pos is not None:
            step_size = self.brush_radius * 2                   # Distance between each fill position.
            travel_vector = (pos[0] - self.prev_brush_pos[0],   # Vector from previous brush position to new one.
                             pos[1] - self.prev_brush_pos[1])   #
            travel_dist = np.linalg.norm(travel_vector)         # Distance between the two points.
            
            # Continue if the distance between the two points is greater than the step size.
            if travel_dist > step_size:
                step_vector = (travel_vector[0] * step_size / travel_dist,  # Vector between each draw position.
                               travel_vector[1] * step_size / travel_dist)  #
                
                # Brush cells at intervals between the previous and new brush position.
                for i in range(1, math.ceil(travel_dist / step_size) + 1):
                    # Draw positions begin at the previous brush position and increment towards the new one.
                    draw_pos = (self.prev_brush_pos[0] + step_vector[0] * i,
                                self.prev_brush_pos[1] + step_vector[1] * i)
                    
                    # Color the cells around the new draw position.
                    self.brush_cells(draw_pos, color)
                
        # Update the stored brush position.
        self.prev_brush_pos = pos
    
    
    def brush_cells(self, pos: tuple, color: int):
        """ Colors the cells around the given brush position.

        Args:
            pos (tuple): Tuple of brush position (float x, float y)
            color (int): Color to fill cells with (0 or 1).
        """
        x_min = math.floor(max(0, pos[0] - self.brush_radius))          # Range of nearby cells to check for brush collision.
        x_max = math.ceil(min(self.width, pos[0] + self.brush_radius))  #
        y_min = math.floor(max(0, pos[1] - self.brush_radius))          #
        y_max = math.ceil(min(self.height, pos[1] + self.brush_radius)) #
        
        # Check each nearby cell for collision with the brush, updating their color when a collision is found.
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if self.pixels[x, y] != color and self.check_brush_collision(cell_pos=(x, y), brush_pos=pos):
                    self.pixels[x, y] = color
    
    
    def check_brush_collision(self, cell_pos: tuple, brush_pos: tuple):
        """ Check if the cell is within the brush radius of the brush position.

        Args:
            cell_pos (tuple): Top left corner of the cell in grid coordinates (int x, int y)
            brush_pos (tuple): Position of brush in grid coordinates (float x, float y)

        Returns:
            bool: True if any point in the cell is within the radius of the brush.
        """
        # Check where the brush center is in relation to the edges of the cell.
        x_idx = bisect([cell_pos[0], cell_pos[0] + 1], brush_pos[0])
        y_idx = bisect([cell_pos[1], cell_pos[1] + 1], brush_pos[1])
        
        # 0 = brush coord is below lower cell coord
        # 1 = brush coord is within cell coord
        # 2 = brush coord is above upper cell coord
        
        # Get the closest point of the cell to the brush center
        x_reference = [cell_pos[0], brush_pos[0], cell_pos[0] + 1][x_idx]
        y_reference = [cell_pos[1], brush_pos[1], cell_pos[1] + 1][y_idx]
    
        # Return True if closest point is within the brush radius
        return math.hypot(x_reference-brush_pos[0], y_reference-brush_pos[1]) <= self.brush_radius
