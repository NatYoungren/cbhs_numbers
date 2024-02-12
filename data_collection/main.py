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
                                #       Using the same size makes it simple to compare datasets.

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
PROMPT_TEXT_SIZE = 42           # Font size of prompt text.


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
            
    return running, reset_canvas, finished_drawing


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


# Unused function.
# This is an example of how to write 'shorter' code.
# Why it is not always better to do this?
def draw_state_but_shorter(surf: pg.Surface, canvas: np.ndarray, brush_size: int):
    """ This is a more compact version of the draw_state function above.
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
        
    # Draws brush as a circle around the mouse.
    pg.draw.circle(surf, (255, 0, 0), pg.mouse.get_pos(), brush_size, 1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                     # TEXT DISPLAY: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def draw_upper_text(surf: pg.Surface, font: pg.font, prompt: str):
    """ Draw prompt text to the upper text region.

    Args:
        surf (pg.Surface): Surface to draw to (presumably the screen).
        font (pg.font): Pygame font with text sizing/styling.
        prompt (str): Text to display.
    """
    draw_text_w_outline(surf=surf,
                        text=prompt,
                        pos=(UPPER_TEXT_POS[0], UPPER_TEXT_POS[1]),
                        font=font, color=TEXT_COLOR, alpha=TEXT_ALPHA,
                        outline_color=(0, 0, 0), outline_alpha=TEXT_ALPHA)


def draw_lower_text(surf: pg.Surface, font: pg.font):
    """ Draw instructional text to the lower text region.

    Args:
        surf (pg.Surface): Surface to draw to (presumably the screen).
        font (pg.font): Pygame font with text sizing/styling.
    """
    draw_text(surf=surf,
              text='Left Click: Draw',
              pos=(LOWER_TEXT_POS[0], LOWER_TEXT_POS[1]-TEXT_SIZE/2*3),
              font=font, color=TEXT_COLOR, alpha=TEXT_ALPHA)
    
    draw_text(surf=surf,
              text='Right Click: Erase',
              pos=(LOWER_TEXT_POS[0], LOWER_TEXT_POS[1]-TEXT_SIZE/2),
              font=font, color=TEXT_COLOR, alpha=TEXT_ALPHA)
    
    draw_text(surf=surf,
              text='SPACE: Next Digit',
              pos=(LOWER_TEXT_POS[0], LOWER_TEXT_POS[1]+TEXT_SIZE/2),
              font=font, color=TEXT_COLOR, alpha=TEXT_ALPHA)    
    
    draw_text(surf=surf,
              text='R: Reset',
              pos=(LOWER_TEXT_POS[0], LOWER_TEXT_POS[1]+TEXT_SIZE/2*3),
              font=font, color=TEXT_COLOR, alpha=TEXT_ALPHA)


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
    """ Draws outlined text by rendering the text multiple times with an offset.
    """
    for x_offset in [-2, 2]:
        for y_offset in [-2, 2]:
            draw_text(surf, text, (pos[0]+x_offset, pos[1]+y_offset), font, outline_color, outline_alpha)
    draw_text(surf, text, pos, font, color, alpha)


def construct_prompt(digit: str='~', adjective: str=''):
    """ Construct an instruction prompt string.

    Args:
        digit (str, optional): Digit to draw. Defaults to '~' placeholder.
        adjective (str, optional): Adjective to use in prompt. Defaults to ''.

    Returns:
        str: Prompt containing digit and adj (adj may be blank).
    """
    prompt = f'Draw a{adjective} number {digit}!'
    return prompt


if __name__ == '__main__':
    main()
