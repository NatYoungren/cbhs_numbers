# For Casco Bay High School
# Nathaniel Alden Homans Youngren
# December 6, 2023

import math
import numpy as np
import pygame as pg

import random
from utils.canvas import Canvas
import utils.torch_helpers as th

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # INSTRUCTIONS: #                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# 1. Hold left click to draw on the canvas.

# 2. Hold right click to erase, or reset the canvas with 'R'.

# 3. Draw as many digits as you like!


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # ALL CONTROLS: #                     #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# m1 -> Draw
# m2 -> Erase

# SPACE -> Randomly shift drawing
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
DRAWN_COLOR = (240, 240, 240)   # Color of drawn cells (R, G, B).
BRUSH_COLOR = (255, 0, 0)       # Color of the brush indicator (R, G, B).
BG_COLOR = (25, 75, 25)         # Color of the background (R, G, B).
                                # NOTE: Seen in the border between cells and any empty space around the grid.

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
    pg.display.set_caption('CBHS - Interactive OCR')

    clock = pg.time.Clock()
    
    # Initialize pygame fonts.
    pg.font.init()
    font = pg.font.Font(TEXT_FONT, TEXT_SIZE)
    prompt_font = pg.font.Font(TEXT_FONT, PROMPT_TEXT_SIZE)

    # Create a dict of OCR models and current predictions.
    ocr_dict = {
                th.load_model('ocr_model.pt'): ('~', 0),
                }
    
    canvas = Canvas(GRID_W, GRID_H) # Initialize a canvas object (see canvas.py).

    running = True              # Flag to continue running the program.

    # Main loop.
    while running:        

        # Handle keypress events (quitting, resetting).
        (running, canvas_changed) = parse_key_input(canvas)
        # NOTE: These flags must be resolved before the next set of events is parsed.
        #   If running = False, the program will exit after this loop finishes.
        
        # Handle mouse events (drawing and erasing).
        canvas_changed = parse_mouse_input(canvas) or canvas_changed

        # Redraw the current state of the canvas.
        draw_canvas(screen, canvas)
        
        # Draw text to the screen.
        if ENABLE_TEXT:
            if not canvas.pixels.any() or canvas.pixels.all():
                ocr_dict = {model:('~', 0) for model in ocr_dict.keys()}
            elif canvas_changed:
                ocr_dict = {model:th.evaluate(model, canvas.pixels) for model in ocr_dict.keys()}

            draw_upper_text(screen, prompt_font, ' - '.join([f'{pred}' for pred, _ in ocr_dict.values()]))
            draw_lower_text(screen, font)
            
        pg.display.flip()
        clock.tick(60)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                   # EVENT HANDLING: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def parse_key_input(canvas: Canvas):
    """ Handle pygame keyboard/window events.
            Events can trigger quit, reset, and save functionality.
    
    Args:
        canvas (Canvas): Canvas object containing current state of the drawing.
    """
    
    # These are the default values we will return each frame unless an event changes them.
    running = True              # By default, keep running the program.
    canvas_changed = False               # By default, do not save the current canvas.

    # Handle pygame events.
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
                canvas.reset()
                canvas.set_brush_radius()
                canvas_changed = True
            
            # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
            # Spacebar randomly shifts the drawing inside the canvas.
            elif event.key == pg.K_SPACE:
                canvas.pixels = th.random_alignment(canvas.pixels)
                canvas_changed = True
        
        # Handle quit event (Control-Q or clicking the X on the window).
        elif event.type == pg.QUIT:
            running = False
            
    return running, canvas_changed


def parse_mouse_input(canvas: Canvas):
    """ Handle pygame mouse inputs.
            Mouse can affect canvas tiles.
    
    Args:
        canvas (Canvas):  Canvas object containing current state of the drawing.
        
    Returns:
        bool: True if the canvas was changed, False otherwise.
    """
    
    # Check which mouse buttons are currently held down.
    held_mouse_buttons = pg.mouse.get_pressed() # NOTE: held_mouse_buttons is a tuple of 3 booleans:
                                                #   held_mouse_buttons[0] is left click.
                                                #   held_mouse_buttons[1] is middle click.
                                                #   held_mouse_buttons[2] is right click.
    
    if any(held_mouse_buttons):
        mouse_pos = pg.mouse.get_pos()                  # Get the current mouse position.
        mouse_x = (mouse_pos[0] - ORIGIN_X) / CELL_W    # Convert pixel coords to cell coords.
        mouse_y = (mouse_pos[1] - ORIGIN_Y) / CELL_H    #
        color = held_mouse_buttons[0]                   # If m1 is held, color is 1, else 0.
        
        canvas.draw(pos = (mouse_x, mouse_y), color = color)
        return True
    
    else:
        # If no mouse buttons are held down, reset the previous brush position.
        canvas.prev_brush_pos = None
        return False

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                  # SCREEN UPDATING: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def draw_canvas(surf: pg.Surface, canvas: Canvas):
    """ Draw the current canvas onto the pygame window.
        Each cell is shown as a colored rectangle.

    Args:
        surf (pygame.Surface): Surface to draw to (presumably the screen).
        canvas (Canvas): Canvas object containing current state of the drawing.
    """
    surf.fill(BG_COLOR) # Used for grid lines between cells and empty border space.
    
    # Width and height of each cell, minus the border on each side.
    width  = CELL_W - BORDER_PX * 2
    height = CELL_H - BORDER_PX * 2
    
    # Iterate over the grid, draw a colored rectangle for each cell.
    for w in range(GRID_W):
        x = ORIGIN_X + CELL_W * w       # Horizontal pixel position of the cell origin (top left corner).
        x += BORDER_PX                  # Add the border width to get the horizontal position of the rectangle we want to draw.

        for h in range(GRID_H):
            y = ORIGIN_Y + CELL_H * h   # Vertical pixel position of the cell origin (top left corner).
            y = y + BORDER_PX           # Add the border width to get the vertical position of the rectangle we want to draw.
            
            rect_vars = (x, y, width, height)   # Arrange the rectangle variables into a tuple.
                                                # NOTE: This format is what pygame.draw.rect() expects.
            
            cell_state = canvas.pixels[w, h]    # Get the state of the current cell.
                                                # NOTE: 0 for empty, 1 for drawn.

            color = [CANVAS_COLOR, DRAWN_COLOR][cell_state] # Set the color based on the cell state.
                                                            # NOTE: This is a neat trick to avoid an if statement.
                                                            #       We can use the state of the cell as an index,
                                                            #       an index of 0 gives us CANVAS_COLOR
                                                            #       and an index of 1 gives us BRUSH_COLOR.
            
            pg.draw.rect(surf, color, rect_vars) # Draw the rectangle.
            
    # Draw brush as a ring around the mouse.
    pg.draw.circle(surf, BRUSH_COLOR, pg.mouse.get_pos(), canvas.brush_radius * min(CELL_W, CELL_H), 1)


# Unused function.
# This is an example of how to write 'shorter' code.
def draw_canvas_but_shorter(surf: pg.Surface, canvas: Canvas):
    """ This is a more compact version of the draw_canvas function above.
            What is one reason to write code like this?
            What is one reason NOT to write code like this?
    
    Args:
        surf (pygame.Surface): Surface to draw to (presumably the screen).
        canvas (Canvas): Canvas object containing current state of the drawing.
    """
    surf.fill(BG_COLOR) # Used for grid lines between cells and empty border space.
    for i, c in enumerate(canvas.pixels.flatten()):
        # Draw each cell as a rectangle, colored for active/inactive.
        pg.draw.rect(surf,
                     [CANVAS_COLOR, DRAWN_COLOR][c],
                     ((ORIGIN_X + CELL_W * (i // canvas.width) + BORDER_PX,
                       ORIGIN_Y + CELL_H * (i % canvas.height) + BORDER_PX,
                       CELL_W - BORDER_PX * 2,
                       CELL_H - BORDER_PX * 2)))
        
    # Draw brush as a ring around the mouse.
    pg.draw.circle(surf, BRUSH_COLOR, pg.mouse.get_pos(), canvas.brush_radius * min(CELL_W, CELL_H), 1)


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#                     # TEXT DISPLAY: #                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def draw_upper_text(surf: pg.Surface, font: pg.font, digit_guess: str):
    """ Draw prompt text to the upper text region.

    Args:
        surf (pg.Surface): Surface to draw to (presumably the screen).
        font (pg.font): Pygame font with text sizing/styling.
        prompt (str): Text to display.
    """
    text_x = UPPER_TEXT_POS[0]
        
    text_y = UPPER_TEXT_POS[1]-PROMPT_TEXT_SIZE/2
    draw_text_w_outline(surf, digit_guess, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA, (0, 0, 0), TEXT_ALPHA)


def draw_lower_text(surf: pg.Surface, font: pg.font):
    """ Draw instructional text to the lower text region.

    Args:
        surf (pg.Surface): Surface to draw to (presumably the screen).
        font (pg.font): Pygame font with text sizing/styling.
    """
    text_x = LOWER_TEXT_POS[0]
    control_text = 'Left Click: Draw'

    text_y = LOWER_TEXT_POS[1]-TEXT_SIZE/2*3
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)
    
    control_text = 'Right Click: Erase'
    text_y = LOWER_TEXT_POS[1]-TEXT_SIZE/2
    draw_text(surf, control_text, (text_x, text_y), font, TEXT_COLOR, TEXT_ALPHA)
    
    control_text = 'R: Reset'
    text_y = LOWER_TEXT_POS[1]+TEXT_SIZE/2
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
    """ Draws outlined text by rendering the text multiple times with an offset.
    """
    for x_offset in [-2, 2]:
        for y_offset in [-2, 2]:
            draw_text(surf, text, (pos[0]+x_offset, pos[1]+y_offset), font, outline_color, outline_alpha)
    draw_text(surf, text, pos, font, color, alpha)


if __name__ == '__main__':
    main()
