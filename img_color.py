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
# Working with paths, file names, directories
from pathlib import Path
import os

class Picture(): 
    def __init__(self, path): 
        # call init with 'path' a Path object
        self.path = path
        self.cluster()

    def cluster(self, n_colors=5): 
        '''
            Perform k-means clustering on the image to determine its dominant colors.
        '''
        self.img = io.imread(self.path)[:, :, :-1]
        pixels = np.float32(self.img.reshape(-1,3))

        n_colors = n_colors
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
        flags = cv2.KMEANS_RANDOM_CENTERS

        # palette gives the chosen cluster centers
        _, labels, self.palette = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)
        # counts gives the size of each cluster. Order is the same as the order of centers in self.palette.
        _, self.counts = np.unique(labels, return_counts=True)
    
    def get_dominant(self): 
        '''
            Return the center (color) of the largest cluster
        '''
        return self.palette[np.argmax(self.counts)]

    def get_distance_to(self, other): 
        '''
            Return the Euclidean distance from the image's dominant color to the user input color. 

            'other' should be a list-like RGB value, ie (233, 0, 1.2)
        '''
        dominant = self.get_dominant()
        return np.linalg.norm(dominant - np.float32(other))
        squared = 0
        for count, value in tuple(dominant): 
            squared += (value - other[count]) ** 2
        return squared ** (1 / 2)

    def plot_palette(self):
        indices = np.argsort(self.counts)[::-1]
        freqs = np.cumsum(np.hstack([[0], self.counts[indices]/float(self.counts.sum())]))
        rows = np.int_(self.img.shape[0]*freqs)

        dom_patch = np.zeros(shape=self.img.shape, dtype=np.uint8)
        for i in range(len(rows) - 1): 
            dom_patch[rows[i]:rows[i+1], :, :] += np.uint8(self.palette[indices[i]])

        fig, ax = plt.subplots()
        ax.imshow(dom_patch)
        ax.axis('off')
        ax.set_title('Dominant colors')
        plt.show()

# for color in dominant_colors: 
#     print(color)

# for picture in pictures:
#     picture.plot_palette()

class Collection():
    def __init__(self, picture_folder='pictures/'): 
        # This must be a directory in the project directory which contains only image files.
        self.picture_folder = picture_folder
        self.pictures = [Picture(path) for path in Path(picture_folder).iterdir()]
        self.dominant_colors = {tuple(picture.get_dominant()):picture for picture in self.pictures}

    def get_closest(self): 
        # Use tkinter color widget to get a color from the user
        color = askcolor()[0]
        min_distance = 1e9
        closest = None
        for key, value in self.dominant_colors.items(): 
            if value.get_distance_to(color) < min_distance: 
                min_distance = value.get_distance_to(color)
                closest = value
        print(min_distance)
        return closest.path

    def plot_palettes():
        pass

collection = Collection()
# print(collection.dominant_colors)
os.startfile(collection.get_closest())
# for picture in collection.pictures: 
#     print(picture.get_dominant())