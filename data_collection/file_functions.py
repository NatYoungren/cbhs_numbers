# For Casco Bay High School
# Nathaniel Alden Homans Youngren
# December 7, 2023

import os
import numpy as np
from PIL import Image
from pathlib import Path
from string import ascii_letters

SAVE_DIR = 'saved_digits'               # The directory in which to save images.
DIGITS = [str(i) for i in range(10)]    # List of digits, saved as strings.
DIGIT_COUNTS = {}                       # Dictionary tracking the number of images in each digit directory.
                                        # NOTE: This is initialized by populate_digit_counts()
                                        #       the first time get_digit() is called
                                        #       and updated by save_canvas().

# List of adjectives to use in prompts/filenames.
ADJECTIVES = [' crooked',
              ' perfect',
              ' careful',
              ' careless',
              ' sloppy',
              ' fancy',
              ' fast',
              ' small',
              ' big',
              ' thin',
              ' wide',
              ' wavy',
              ' jagged',
              ' tall',
              ' short',
              ' leaning',
              'n upright',
              ' normal',
              ' weird',
              ' standard',
              ' strange']

ACTIVE_CELL_COLOR   = (255, 255, 255)   # Color of drawn cells (R, G, B) is white.
INACTIVE_CELL_COLOR = (000, 000, 000)   # Color of blank cells (R, G, B) is black.

ALLOWED_CHARACTERS = ascii_letters      # Characters allowed in filename elements.

# Returns a digit with the fewest images.
def get_digit() -> str:
    # If we haven't populated DIGIT_COUNTS yet, do so now.
    if len(DIGIT_COUNTS) == 0:
        populate_digit_counts()
    
    min_index = np.argmin(list(DIGIT_COUNTS.values()))
    return DIGITS[min_index]

# Returns a random adjective.
# NOTE: Returns no adjective (blank string) with probability skip_chance.
def get_adjective(skip_chance: float=0.5) -> str:
    if np.random.random() < skip_chance:
        return ''
    return ADJECTIVES[np.random.randint(len(ADJECTIVES))]

# Populates DIGIT_COUNTS with the number of images in each digit directory.
def populate_digit_counts(target_dir: str=SAVE_DIR) -> None:
    for d in DIGITS:
        digit_dir = os.path.join(target_dir, d)
        Path(digit_dir).mkdir(parents=True, exist_ok=True)
        DIGIT_COUNTS[d] = len(os.listdir(digit_dir))

# Saves a canvas as a png to the SAVE_DIR.
def save_canvas(canvas: np.ndarray,
                digit: str,
                adjective: str='',
                target_dir: str=SAVE_DIR,
                img_ext='png') -> None:
    
    # Remove all non-alphabetic characters from adjective (just in case).
    adjective = whitelist_chars(adjective.lower(), ALLOWED_CHARACTERS)
    
    # If adjective is blank, replace with a unique placeholder.
    if len(adjective) == 0:
        adjective = '$NOADJ'
        
    # Create the digit directory if it doesn't exist.
    digit_dir = os.path.join(target_dir, digit)
    Path(digit_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate the filename template (e.g. '0_$NOADJ_{}.png').
    # NOTE: The '{}' will be replaced with the first available image number.
    filename = f'{digit}_{adjective}_{"{}"}.{img_ext}'

    # Find the next available image number.
    # NOTE: Searching like this is not scalable, but this is fine for limited numbers of files.
    img_num = 0
    while os.path.exists(os.path.join(digit_dir, filename.format(img_num))):
        img_num += 1
    
    # Convert the canvas to a PIL image and save it.
    img = img_frombytes(canvas)
    img.save(os.path.join(digit_dir, filename.format(img_num)))
    
    # Update DIGIT_COUNTS to reflect the new digit.
    DIGIT_COUNTS[digit] += 1



# Converts a numpy array of 0s and 1s to a PIL image.
def img_frombytes(data: np.ndarray) -> Image:
    # Our array and our image have different row/column orders.
    # The fix is to transpose the array (with '.T') before converting to bytes.
    size = data.shape
    data = data.T.copy(order='C')
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

# Removes all characters not in allowed_characters from raw_string.
def whitelist_chars(raw_string: str, allowed_characters: str) -> str:
    return ''.join([c for c in raw_string if c in allowed_characters])
