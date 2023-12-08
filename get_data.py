from quickdraw import QuickDrawDataGroup, QuickDrawData
from pathlib import Path
import os
from doodle.params import *

image_size = (64, 64)

def generate_class_images(name, max_drawings, recognized):
    directory = Path("data_50_a/" + name)
    if not directory.exists():
        directory.mkdir(parents=True)

    images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
    for img in images.drawings:
        filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
        img.get_image(stroke_width=3).resize(image_size).save(filename)

for label in QuickDrawData().drawing_names[:50]:
    generate_class_images(label, max_drawings=10000, recognized=True)
