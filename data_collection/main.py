# For Casco Bay High School
# Nathaniel Alden Homans Youngren
# December 6, 2023

import time
import math
import numpy as np
import pygame as pg

import file_functions as ff

# TODO: Move into helper_functions.py or remove completely.
# # All non-main code is in helper_functions.py
# # This is to keep the main.py file clean and readable.
# import helper_functions as hf


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # INSTRUCTIONS: #                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


# 1. Hold left click to draw on the canvas, following the prompt in the window title.

# 2. Hold right click to erase, or reset the canvas with 'R'.

# 3. Press Spacebar when you are finished drawing.

# 4. Draw as many digits as you like!


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # ALL CONTROLS: #                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# m1 -> Draw
# m2 -> Erase

# SPACE -> Save and continue
# 'R' -> Reset

# ESC -> Quit


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                      # SETTINGS: #                      #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# TEXT REGION CONSTANTS
ENABLE_TEXT = True              # If True, text-input regions will be added to the top and bottom of the window.
TEXT_REGION_H = 100             # Height of the text-input region (in pixels).

# CANVAS CONSTANTS
CANVAS_W, CANVAS_H = 560, 560   # Size of the canvas region (in pixels).
                                # NOTE: Height of the pygame window is CANVAS_H + TEXT_REGION_H*2.
                                
GRID_W, GRID_H = 28, 28         # Determines the size of the drawing canvas (in cells).
                                # NOTE: This determines the size of any saved images (in pixels).
                                #       The well-known MNIST dataset of handwritten digits uses 28x28 images.
                                #       Using the same size makes it simple to compare or mix datasets.

SQUARE_CELLS = True             # If True, cells will always be square even when the window is not.

BORDER_PX = 1                   # Width of the border around each cell (in pixels).
                                # NOTE: Because this border is drawn on both sides of each cell,
                                #       the total border width is 2*BORDER_PX.

# COLOR CONSTANTS
TEXT_COLOR = (255, 255, 255)    # Color of text (R, G, B).
CANVAS_COLOR = (0, 0, 0)        # Color of un-drawn cells (R, G, B).
BRUSH_COLOR = (240, 240, 240)   # Color of drawn cells (R, G, B).
BG_COLOR = (25, 75, 25)         # Color of the background (R, G, B).
                                # NOTE: Seen in the border between cells and any empty space around the grid.

# BRUSH CONSTANTS
BRUSH_SCALE = 1.25              # Radius of the brush (in cells).
BRUSH_SCALE_RANGE = [0.5, 1.2]  # Range of possible brush scales (in cells).
RANDOM_BRUSH_SCALE = True       # If True, brush scale will be randomly selected from BRUSH_SCALE_RANGE.

# TEXT CONSTANTS
TEXT_FONT = 'freesansbold.ttf'  # Name of text font.
TEXT_ALPHA = 255                # Transparency value of text from 0 to 255.
TEXT_SIZE = 24                  # Font size.
PROMPT_TEXT_SIZE = 36           # Font size of prompt text.


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                        # SETUP: #                       #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

TEXT_REGION_H *= ENABLE_TEXT                        # If ENABLE_TEXT is False, TEXT_REGION_H will be set to 0.

SCREEN_W = CANVAS_W                                 # Width of the pygame window (in pixels).
SCREEN_H = CANVAS_H + TEXT_REGION_H * 2             # Height of the pygame window (in pixels).

UPPER_TEXT_REGION = pg.Rect(0,                      # Region where prompt text is displayed.
                            0,                      # NOTE: This will be a strip at the top of the pygame window.
                            SCREEN_W,
                            TEXT_REGION_H)      

LOWER_TEXT_REGION = pg.Rect(0,                      # Region where instructional text is displayed.
                            SCREEN_H-TEXT_REGION_H, # NOTE: This will be a strip at the bottom of the pygame window.
                            SCREEN_W,
                            TEXT_REGION_H)

CELL_W = CANVAS_W / GRID_W                          # Size of each canvas cell (in pixels).
CELL_H = CANVAS_H / GRID_H

if SQUARE_CELLS:
    CELL_W = CELL_H = min(CELL_W, CELL_H)           # If SQUARE_CELLS, set cell size to be the same in both dimensions.

ORIGIN_X = (CANVAS_W - CELL_W * GRID_W) / 2         # Calculate the origin of the canvas (top left corner of the top left cell).
ORIGIN_Y = (CANVAS_H - CELL_H * GRID_H) / 2         # NOTE: We use this to center the canvas regardless of screen shape.

ORIGIN_Y += TEXT_REGION_H                           # Offset the canvas vertically by the height of the upper text region.

UPPER_TEXT_POS = (SCREEN_W/2, TEXT_REGION_H/2)      # To center other text, get the text regions centers (in pixels).
LOWER_TEXT_POS = (SCREEN_W/2, SCREEN_H-TEXT_REGION_H/2)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                    # MAIN FUNCTION: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def main():
    
    # Initialize pygame window.
    pg.init()
    screen = pg.display.set_mode((SCREEN_W, SCREEN_H))
    pg.display.set_caption('CBHS - Handwritten Digits')

    clock = pg.time.Clock()
    
    # Initialize pygame fonts.
    pg.font.init()
    font = pg.font.Font(TEXT_FONT, TEXT_SIZE)
    prompt_font = pg.font.Font(TEXT_FONT, PROMPT_TEXT_SIZE)

    # Generate a list of rectangles, one for each cell in the grid.
    # Each rectangle is a tuple of (x, y, width, height).
    cell_rects = generate_rects(origin=(ORIGIN_X, ORIGIN_Y),
                                cell_shape=(CELL_W, CELL_H),
                                grid_shape=(GRID_W, GRID_H))

    brush_size = generate_brush_size()  # Generate brush size (in pixels).
    
    digit = ff.get_digit()          # Digit to use in prompt.
    adjective = ff.get_adjective()  # Adjective to use in prompt.
    
    canvas = None               # 2D Numpy array holding the state of each cell, 0 for empty, 1 for drawn.
                                # NOTE: This is where the current drawing is stored.
                                #       We will reset this to a blank array whenever we begin a drawing.
    
    finished_drawing = False    # Flag to move on to the next drawing.
    
    running = True              # Flag to continue running the program.
    reset_canvas = True         # Flag to reset the canvas.
                                # NOTE: This starts as True, as we want to reset the canvas on the first frame.

    # Main loop.
    while running:
        
        # If flagged, save the current canvas, select a new digit, trigger a canvas reset.
        if finished_drawing:
            # If the canvas is all blank or all drawn, don't save it.
            if canvas.any() and not canvas.all():
                ff.save_canvas(canvas, digit=digit, adjective=adjective)
                digit = ff.get_digit()              # Get a new digit.
                adjective = ff.get_adjective()      # Get a new adjective.
                brush_size = generate_brush_size()  # Get a new brush size (if RANDOM_BRUSH_SCALE is True).
                reset_canvas = True                 # Set a flag to reset the canvas.
        
        # If flagged, reset the canvas.
        if reset_canvas:
            canvas = np.zeros((GRID_W, GRID_H), dtype=int)  # Set all cells to 0 (empty).
            reset_canvas = False                            # Flag has been resolved, reset it.
        
        # Handle input events.
        (running, reset_canvas, finished_drawing) = parse_events(canvas, cell_rects, brush_size)
        # NOTE: These flags must be resolved before the next set of events is parsed.
        #   If running = False, the program will exit after this loop finishes.
        #   If reset_canvas = True, the canvas will be reset at the beginning of the next loop.
        #   If finished_drawing = True, the image will be saved and the canvas reset with a new digit at the start of the next loop.
        
        
        # Redraw the current state of the canvas
        draw_state(screen, canvas, brush_size)
        if ENABLE_TEXT:
            draw_upper_text(screen, prompt_font, construct_prompt(digit, adjective))
            draw_lower_text(screen, font)
        else:
            # If we are not drawing text to the screen, put the prompt in the window title.
            pg.display.set_caption(construct_prompt(digit, adjective))

        # Update the screen
        pg.display.flip()
        clock.tick(60)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # EVENT HANDLING: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def parse_events(canvas: np.ndarray, rects: np.ndarray, brush_size: int, allow_drawing: bool=True):
    """ Handle pygame events.
            Keypresses generally change the program state.
            Mouse buttons generally effect the drawing state.
    
    Args:
        canvas (np.ndarray(int)): 2D Numpy array holding the state of each canvas cell, 0 for empty, 1 for drawn.
        rects (np.ndarray(int)): 2D Numpy array holding the pixel corrdinates of each canvas cell, (x, y, w, h).
        brush_size (int): Radius of the brush in pixels.
        allow_drawing (bool, optional): If True, allow drawing with the mouse. Defaults to True.
    """
    
    # These are the default values we will return each frame unless an event changes them.
    running = True              # By default, keep running the program.
    reset_canvas = False        # By default, do not reset the canvas.
    finished_drawing = False    # By default, do not move on to the next drawing.
    change_text = False         # By default, do not get new text input.
    
    # Handle input events.
    for event in pg.event.get():

        # Handle keypresses
        if event.type == pg.KEYDOWN:
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # Escape key quits after resolving the current frame.
            if event.key == pg.K_ESCAPE:
                running = False
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
            # R key resets the current canvas.
            elif event.key == pg.K_r:
                reset_canvas = True
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Spacebar moves on to the next drawing.
            elif event.key == pg.K_SPACE:
                finished_drawing = True
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Enter triggers name update.
            elif event.key == pg.K_RETURN and ENABLE_NAMES:
                change_text = True
        
        # Handle quit event (Control-Q or clicking the X on the window).
        elif event.type == pg.QUIT:
            running = False
    
    # If drawing is allowed, handle mouse events.
    if allow_drawing:
        
        # Find out which mouse buttons are currently held down.
        held_mouse_buttons = pg.mouse.get_pressed() # NOTE: held_mouse_buttons is a tuple of 3 booleans:
                                                    #   held_mouse_buttons[0] is left click.
                                                    #   held_mouse_buttons[1] is middle click.
                                                    #   held_mouse_buttons[2] is right click.
        
        # If any mouse buttons are held down...
        if any(held_mouse_buttons):
            
            # Get a list of all cells that are within BRUSH_SIZE pixels of the mouse position.
            brushed_cells = get_brushed_cells(pg.mouse.get_pos(), rects, brush_size)
            
            # We can use the state of the left mouse button as the state of our 'brush'.
            brush_state = held_mouse_buttons[0]
            
            # Set all brushed cells to match the state of the brush.
            for clicked_cell in brushed_cells:
                canvas[clicked_cell[0], clicked_cell[1]] = brush_state
                
            # NOTE: This is a neat trick to avoid an if statement. ^^
            #       This works because we already know that at least one mouse button is being held down.
            
            #       The state of the left mouse button ( held_mouse_buttons[0] ) is either True or False (0 or 1).
            
            #       If the left button is held, brush_state = True, and we set all brushed cells to 1.  (drawing)
                        
            #       If left mouse button is NOT held, that means the right or middle buttons must be.
            #       In this situation, brush_state = False, and we set all brushed cells to 0. (erasing)
            
    return running, reset_canvas, finished_drawing, change_text


def parse_name_events(name: str):
    """ Handle pygame events while updating the name.
            Backspace removes the last letter.
            All allowed unicode characters are added to the name.
            Enter ends the name update.
    Args:
        name (str): The current name.

    Returns:
        (bool, bool, str): Flag to continue running,flag to finish updating name, and the new name string.
    """
    # These are the default values we will return each frame unless an event changes them.
    running = True
    update_name = True
    
    # Handle input events.
    for event in pg.event.get():

        # Handle keypresses
        if event.type == pg.KEYDOWN:
                
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # Escape key quits after resolving the current frame.
                if event.key == pg.K_ESCAPE:
                    running = False
                
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # Enter ends the name update.
                elif event.key == pg.K_RETURN:
                    update_name = False
                
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # Backspace removes the last letter.
                elif event.key == pg.K_BACKSPACE:
                    name = name[:-1]
                    
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # All other allowed keys are added to the name.
                else:
                    if len(name) < NAME_LENGTH_LIMIT:
                        new_char = event.unicode
                        if new_char in ff.ALLOWED_CHARACTERS:
                            name += new_char
                    
        # Handle quit event (Control-Q or clicking the X on the window).
        elif event.type == pg.QUIT:
            running = False
            
    return running, update_name, name
    

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                  # SCREEN UPDATING: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def draw_state(surf: pg.Surface, canvas: np.ndarray, brush_size: int):
    """ Draw the current canvas onto the pygame window.
        Each cell is shown as a colored rectangle.

    Args:
        surf (pygame.Surface): Surface to draw to (presumably the screen).
        canvas (np.ndarray(int)): 2D Numpy array holding the state of each cell, 0 for empty, 1 for drawn.
    """
    surf.fill(BG_COLOR) # Used for grid lines between cells and empty border space.
    
    # Width and height of each cell, minus the border on each side.
    width  = CELL_W - BORDER_PX*2
    height = CELL_H - BORDER_PX*2
    
    # Iterate over the grid, draw a colored rectangle for each cell.
    for w in range(GRID_W):
        x = ORIGIN_X + CELL_W * w       # Horizontal pixel position of the cell origin (top left corner).
        x += BORDER_PX                  # Add the border width to get the horizontal position of the rectangle we want to draw.

        for h in range(GRID_H):
            y = ORIGIN_Y + CELL_H * h   # Vertical pixel position of the cell origin (top left corner).
            y = y + BORDER_PX           # Add the border width to get the vertical position of the rectangle we want to draw.
            
            rect_vars = (x, y, width, height)   # Arrange the rectangle variables into a tuple.
                                                # NOTE: This format is what pygame.draw.rect() expects.
            
            cell_state = canvas[w, h]           # Get the state of the current cell.
                                                # NOTE: 0 for empty, 1 for drawn.

            color = [CANVAS_COLOR, BRUSH_COLOR][cell_state] # Set the color based on the cell state.
                                                            # NOTE: This is a neat trick to avoid an if statement.
                                                            #       We can use the state of the cell as an index,
                                                            #       an index of 0 gives us CANVAS_COLOR
                                                            #       and an index of 1 gives us BRUSH_COLOR.
            
            pg.draw.rect(surf, color, rect_vars) # Draw the rectangle.
            
    # Draws brush as a circle around the mouse.
    pg.draw.circle(surf, (255, 0, 0), pg.mouse.get_pos(), brush_size, 1)

# Unused function. This is an example of how to write 'shorter' code, and why it is not always a good thing.
def draw_state_but_shorter(surf: pg.Surface, canvas: np.ndarray):
    """ This is a much 'shorter' version of the draw_state function above. (minus the brush)
            What is one reason to write code like this?
            What is one reason NOT to write code like this?
    
    Args:
        surf (pygame.Surface): Surface to draw to (presumably the screen).
        canvas (np.ndarray(int)): 2D Numpy array holding the state of each cell, 0 for empty, 1 for drawn.
    """
    surf.fill(BG_COLOR) # Used for grid lines between cells and empty border space.
    for i, c in enumerate(canvas.flatten()):
        # Draw each cell as a rectangle, colored for active/inactive.
        pg.draw.rect(surf,
                     [CANVAS_COLOR, BRUSH_COLOR][c],
                     ((ORIGIN_X + CELL_W * (i//canvas.shape[0]) + BORDER_PX,
                       ORIGIN_Y + CELL_H * (i % canvas.shape[1]) + BORDER_PX,
                       CELL_W - BORDER_PX*2,
                       CELL_H - BORDER_PX*2)))


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                     # TEXT DISPLAY: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def draw_upper_text(surf: pg.Surface, font: pg.font, prompt_font: pg.font, name:str, prompt: str):
    # Upper Instructions:
    #   Current prompt
    #   Name change controls

    text_x = UPPER_TEXT_POS[0]
        
    text_y = UPPER_TEXT_POS[1]-PROMPT_TEXT_SIZE/2
    draw_text_w_outline(surf, prompt, (text_x, text_y), prompt_font, TEXT_COLOR, TEXT_ALPHA, (0, 0, 0), TEXT_ALPHA)

    if ENABLE_NAMES:
        control_text = f'ENTER: Set Your Name!' 
        
        if len(name) > 0: # If there is currently a name, display it.
            text_y = UPPER_TEXT_POS[1]+TEXT_SIZE/2
            draw_text(surf, f'Name: {name}', (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)
            control_text = f'ENTER: Change Your Name' # NOTE: Overrides the control text if a name is set.
        
        text_y = UPPER_TEXT_POS[1]+TEXT_SIZE/2*3
        draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)


def draw_lower_text(surf: pg.Surface, font: pg.font):
    # Lower Controls:
    #   Left click
    #   Right click
    #   Space
    #   R

    text_x = LOWER_TEXT_POS[0]
    control_text = 'Left Click: Draw'

    text_y = LOWER_TEXT_POS[1]-TEXT_SIZE/2*3
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)
    
    control_text = 'Right Click: Erase'
    text_y = LOWER_TEXT_POS[1]-TEXT_SIZE/2
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)
    
    control_text = 'SPACE: Next Digit'
    text_y = LOWER_TEXT_POS[1]+TEXT_SIZE/2
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)
    
    control_text = 'R: Reset'
    text_y = LOWER_TEXT_POS[1]+TEXT_SIZE/2*3
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)


def draw_name_prompt(surf: pg.Surface, font: pg.font, prompt_font: pg.font, name: str):
    surf.fill(BG_COLOR) # Used for grid lines between cells and empty border space.
    text_x = SCREEN_CENTER[0]
    text_y = SCREEN_CENTER[1]-PROMPT_TEXT_SIZE/2*3
    draw_text_w_outline(surf, f'Type Your Name:',  (text_x, text_y), prompt_font, TEXT_COLOR, TEXT_ALPHA, (0, 0, 0), TEXT_ALPHA)
    
    text_y = SCREEN_CENTER[1]-PROMPT_TEXT_SIZE/2
    
    postfix = '_' if len(name) < NAME_LENGTH_LIMIT and time.time() % 1 > 0.5 else '  '
    draw_text_w_outline(surf, f'{name}{postfix}',  (text_x, text_y), prompt_font, TEXT_COLOR, TEXT_ALPHA, (0, 0, 0), TEXT_ALPHA)

    
    control_text = f'ENTER: Continue'
    text_y = SCREEN_CENTER[1]+TEXT_SIZE/2*3
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)


def draw_text(surf, text, pos, font, color, alpha):
    """ Blit text onto the surface at the given position.

    Args:
        surf (pygame.Surface): Surface to draw text to.
        text (str): Text to draw.
        pos (int, int): Coords at which to center the text.
        font (pygame.font.Font): Font to use for text.
        color (int, int, int): RGB color of text.
        alpha (int): Alpha value of text from 0 to 255.
    """
    text_surface = font.render(text, True, color)
    text_surface.set_alpha(alpha)
    rect = text_surface.get_rect()
    rect.center = pos
    surf.blit(text_surface, rect)


def draw_text_w_outline(surf, text, pos, font, color, alpha, outline_color, outline_alpha):
    for x_offset in [-2, 2]:
        for y_offset in [-2, 2]:
            draw_text(surf, text, (pos[0]+x_offset, pos[1]+y_offset), font, outline_color, outline_alpha)
    draw_text(surf, text, pos, font, color, alpha)


def construct_prompt(digit: str='~', adjective: str='', name: str=''):
    """ Construct an instruction prompt string.

    Args:
        digit (str): Digit to draw.
        adjective (str): Adjective to use in prompt.
        name (str): Name to use in prompt.

    Returns:
        str: Prompt containing digit, adj, and possibly name..
    """
    
    # if len(name) > 0:
    #     prompt = f'{name}, draw a{adjective} number {digit}!'
    # else:
    prompt = f'Draw a{adjective} number {digit}!'
    return prompt

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # CANVAS/DRAWING: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def generate_rects(origin: tuple, cell_shape: tuple, grid_shape: tuple):
    """ Generates an array of rectangles.
        Each rectangle is 4 integers: (x, y, width, height).
        There is rectangle for each cell in the grid.
        
        These rectangles are used to determine which grid cells intersect with the brush.
        Look at the get_cells and brush_collision functions for more info.
        
    Args:
        origin (tuple):     Origin of the grid  (x, y of top left corner) in pixels
        cell_shape (tuple): Shape of each cell  (width, height) in pixels
        grid_shape (tuple): Shape of the grid   (width, height) in cells

    Returns:
        np.ndarray: 2D Numpy array of rectangles.
    """
    rect_corners = np.empty((grid_shape[0], grid_shape[1], 4), dtype=int)
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            # Our X and Y values are: edge of canvas + (cell index * cell size)
            rect_corners[x, y] = [origin[0] + (x * cell_shape[0]),  # X
                                  origin[1] + (y * cell_shape[1]),  # Y
                                  cell_shape[0],                    # Width
                                  cell_shape[1]]                    # Height
    return rect_corners


def get_brushed_cells(pos, canvas_rects, brush_size):
    """ Return all cells within brush_size pixels of the clicked screen position.

    Args:
        pos (int, int): Clicked screen position.
        canvas_rects
        brush_size (int): Size of the brush in pixels.

    Returns:
        list: List of cell coordinates within the brush area.
    """
    # This currently checks every cell in the canvas.
    # We could make this faster by only checking cells which we know are close to the brush area.
    cells = []
    s = canvas_rects.shape
    for x in range(s[0]):
        for y in range(s[1]):
            if brush_collision(canvas_rects[x, y], (pos[0], pos[1], brush_size)):
                cells.append((x, y))
    return cells

# Original: https://stackoverflow.com/questions/24727773/detecting-rectangle-collision-with-a-circle
# NOTE: (Fixed issue in original where rect width/height were halved unnecessarily)
def brush_collision(rect_cell: tuple,      # Rectangle (x, y, width, height)
                    circle_brush: tuple):  # Circle (x, y, radius)
    """ Returns True if there is any overlap between a rectangle and a circle.
        Used to determine which cells are within the brush area.

    Args:
        rect_cell (x, y, w, h): A rectangle, where (x, y) is the top left corner, and (w, h) is the width and height.
        circle_brush (x, y, r): A circle, where (x, y) is the center, and r is the radius.

    Returns:
        bool: True if the rectangle and circle overlap, False otherwise.
    """

    # Rectangle coords are the top left corner and width/height (x, y, w, h).
    r_left =    rect_cell[0]
    r_top =     rect_cell[1]
    r_width =   rect_cell[2]
    r_height =  rect_cell[3]
    
    # Circle coords are the center and radius (x, y, r).
    c_center_x = circle_brush[0]
    c_center_y = circle_brush[1]
    c_radius =   circle_brush[2]
    
    # Use the width and height to calculate the bottom right corner of our rect.
    r_right =   r_left + r_width
    r_bottom =  r_top + r_height
    
    #       r_left, r_top ------ r_right, r_top
    #           |                       |
    #           |                       |
    #           |                       |
    #           |                       |
    #       r_left, r_bottom -- r_right, r_bottom

    # Use the radius to calculate a rectangular bounding-box around our circle.
    c_left =     c_center_x - c_radius
    c_top =      c_center_y - c_radius
    c_right =    c_center_x + c_radius
    c_bottom =   c_center_y + c_radius
    
    #               ._.     
    #               |0|
    #               .-.
    
    # If our rect and bounding box do not overlap in any way, there is no collision.
    if r_right < c_left or r_left > c_right or r_bottom < c_top or r_top > c_bottom:
        return False

    # Since the circle could be smaller than the rectangle, we check if the circle is inside the rectangle.
    if r_left <= c_center_x <= r_right and r_top <= c_center_y <= r_bottom:
        return True
    
    # We can now check if any of the corners of the rectangle are inside the circle.
    for x in (r_left, r_right):
        for y in (r_top, r_bottom):
            
            # If the distance between circle center and rect corner is less than circle radius, there is a collision.
            if math.hypot(x-c_center_x, y-c_center_y) <= c_radius:
                return True
    
    # NOTE: Can you figure out when the above check will fail to identify an overlap?
    #       It isn't a big deal for our purposes, but it could matter a lot if this collision code was used elsewhere.

    return False


def generate_brush_size():
    """ Generate a brush size in pixels.
        If RANDOM_BRUSH_SCALE is True, a brush scale will be randomly selected from BRUSH_SCALE_RANGE.
        Otherwise, the brush scale will be set to BRUSH_SCALE.
        Brush size = cell size * brush scale.

    Returns:
        int: Radius of the brush in pixels.
    """
    if RANDOM_BRUSH_SCALE:
        brush_scale = np.random.uniform(*BRUSH_SCALE_RANGE)
    else:
        brush_scale = BRUSH_SCALE
        
    return int(round(min((CELL_W, CELL_H)) * brush_scale))


if __name__ == '__main__':
    main()
