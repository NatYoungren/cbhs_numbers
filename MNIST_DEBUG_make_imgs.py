# Description: This script can read the mnist_~~~.csv file sand convert each row into an image

from PIL import Image
import numpy as np
from pathlib import Path
import os

filename = 'mnist_test.csv'
out_dir = 'mnist_test_imgs'

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

def row_to_image(row):
    pixels = row.split(',')
    pixels = [int(p) for p in pixels]
    label = pixels[0]
    pixels = np.array(pixels[1:])
    pixels.reshape((28, 28))
        
    img = Image.new('L', (28, 28))
    img.putdata(pixels)
    return label, img

with open(filename, 'r') as f:
    for i, row in enumerate(f):
        label, img = row_to_image(row)
        out_path = os.path.join(out_dir, str(label))
        Path(out_path).mkdir(parents=True, exist_ok=True)
        img.save(os.path.join(out_path, f'img_%05d.png' % i))
