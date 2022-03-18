'''
    Created 3-14-22.  Find the dominant colors of images in a directory.  Find the image 
    with the dominant color closest to the user selected color.

    Dominant colors found by k-means clustering following user Tonechas at:
    https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
'''


# GUI
import tkinter as tk
from tkinter import filedialog
# Image processing
import numpy as np
from skimage import io
# Working with paths, file names, directories
from pathlib import Path, PureWindowsPath
import os

filestring = "C:/Users/whomola/Pictures/Camera Roll/small_test/green.jpg"

# root = tk.Tk()
# Stop an empty root window from appearing
tk.Tk().withdraw()
picture_folder = filedialog.askdirectory()
# root.destroy()
picture_path = Path(picture_folder)

pictures = [path for path in picture_path.iterdir()]

for picture in pictures:
    print(str(picture))
    print(picture.as_posix())

with open('test.txt', 'w') as file:
    for picture in pictures:
        file.write(' ; '.join([picture.as_posix(), 'wer we', 'sdlfkj  s']))

# print(str(path))
# print(path.as_posix())

