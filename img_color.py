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
import matplotlib.pyplot as plt
import numpy as np
# Image processing
import cv2
from skimage import io

# Use tkinter color widget to get a color from the user
# color = askcolor()
# print(color)

img = io.imread('blue.png')[:, :, :-1]
pixels = np.float32(img.reshape(-1,3))

n_colors = 5
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
flags = cv2.KMEANS_RANDOM_CENTERS

# palette gives the chosen cluster centers
_, labels, palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
# counts gives the size of each cluster. Order is the same as the order of centers in palette.
_, counts = np.unique(labels, return_counts=True)

print(palette[np.argmax(counts)])

indices = np.argsort(counts)[::-1]
freqs = np.cumsum(np.hstack([[0], counts[indices]/float(counts.sum())]))
rows = np.int_(img.shape[0]*freqs)

dom_patch = np.zeros(shape=img.shape, dtype=np.uint8)
for i in range(len(rows) - 1): 
    dom_patch[rows[i]:rows[i+1], :, :] += np.uint8(palette[indices[i]])

fig, ax = plt.subplots()
ax.imshow(dom_patch)
ax.axis('off')
ax.set_title('Dominant colors')
plt.show()