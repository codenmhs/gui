'''
    Created 3-14-22.  Find the dominant colors of images in a directory.  Find the image 
    with the dominant color closest to the user selected color.

    Dominant colors found by k-means clustering following user Tonechas at:
    https://stackoverflow.com/questions/43111029/how-to-find-the-average-colour-of-an-image-in-python-with-opencv
'''


# GUI
import tkinter as tk
from tkinter.colorchooser import askcolor
from tkinter import filedialog
# Plotting
import matplotlib
# Necessary to get pyplot and tk to play well.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# Image processing
import numpy as np
import cv2
from skimage import io
# Working with paths, file names, directories
from pathlib import Path
import os

class Picture(): 
    def __init__(self, path=None, stats=None):
        if stats: 
            # If there is a string of existing image data, use it to populate the Picture's fields
            strings = stats.split(';')
            self.path = Path(strings[0])
            self.counts = np.fromstring(self.strip_brackets(strings[3]), sep=',', dtype=int)
            self.palette = np.array([np.fromstring(self.strip_brackets(row), sep=',') for row in strings[2].split(',\n')])
            self.img = io.imread(self.path)[:, :, :-1]
        else: 
            # call init with 'path' a Path object
            self.path = path
            self.cluster()

    @staticmethod
    def strip_brackets(string):
        '''
            Strip square brackets for reading numpy exported strings back into numpy arrays.
        '''
        return string.replace('[[','').replace(']]','').replace('[','').replace(']','')

    def __repr__(self):
        # Stop numpy printing in scientific notation, see https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list
        np.set_printoptions(suppress=True)
        return ' ; '.join([
            str(self.path), 
            # The center of the largest color cluster
            np.array2string(self.get_dominant(), precision=3, floatmode='fixed', max_line_width=None, separator=',').replace('\n', '').replace(' ', ''),
            # An array of all color centers, each a three number array.
            np.array2string(self.palette, precision=3, floatmode='fixed', max_line_width=None, separator=',').replace(' ', ''), 
            # The size of the clusters around the centers above, in same order.
            np.array2string(self.counts, precision=3, floatmode='fixed', max_line_width=None, separator=',').replace('\n', '')]).replace(' ', '')

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
        ax.set_title(str(self.path))
        plt.show()

class Collection():
    def __init__(self, choose_dir=False, picture_folder='pictures/'): 
        # picture_folders must be a directory which contains only image files.
        if choose_dir:
            root = tk.Tk()
            root.update()
            # Stop an empty root window from appearing
            # tk.Tk().withdraw()
            picture_folder = filedialog.askdirectory()
            root.destroy()
        self.picture_folder = Path(picture_folder)

        picture_data = self.picture_folder / 'picture_data.txt'
        self.pictures = []
        if picture_data.is_file():
            # If a file of image stats already exists, use it to instantiate the Picture objects instead of computing anew.
            text = []
            with picture_data.open() as file: 
                for line in file.read().split('\n\n'):
                    if line: 
                        # Since the last picture's data has a \n\n after it, we need to check that
                        # we are not trying to instantiate Picture with the empty string at the end of the split array.
                        self.pictures.append(Picture(stats=line))
        else: 
            self.pictures = [Picture(path=path) for path in self.picture_folder.iterdir()]
            self.export_colors()
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
        os.startfile(closest.path)
        return closest.path

    def export_colors(self, output_file='picture_data.txt'): 
        with open(self.picture_folder / output_file, 'w') as file: 
            for picture in self.pictures: 
                file.write(picture.__repr__() + '\n\n')

    def plot_palettes():
        pass

collection = Collection(True)
# collection.export_colors()
collection.pictures[0].plot_palette()
# Open the image with the closest color
# os.startfile(collection.get_closest())