'''
    Created 3-14-22.  Find the dominant colors of images in a directory.  Find the image 
    with the dominant color closest to the user selected color.

    Dominant colors found by k-means clustering following user Tonechas at:
    https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
'''


# GUI
import tkinter as tk
from tkinter.colorchooser import askcolor
# Plotting
import matplotlib
# Necessary to get pyplot and tk to play well.
matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
import numpy as np
# Image processing
import cv2
from skimage import io


root = tk.Tk()
root.title("Choose Color")
root.geometry('300x300')

def change_color():
    colors = askcolor(title="Choose a Color")
    root.configure(bg=colors[1])

tk.Button(root, text="Change background color", command=change_color).pack(expand=True)

root.mainloop()