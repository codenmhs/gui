''' Follows http://physicalmodelingwithpython.blogspot.com/2016/04/make-your-own-gui-with-python.html
Created 3-14-22'''

import tkinter as tk
import matplotlib
# Necessary to get pyplot and tkinter to play well.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Create the top level window object
root = tk.Tk()

# Add a text box
welcome = tk.Label(text="Greetings, user.")
welcome.pack()

# A close-window function
def quit(event=None): 
    root.destroy()

# Bind close-window function to escape keep event.
root.bind('<Escape>', quit)

exit_button = tk.Button(text="Exit", command=quit)
exit_button.pack(side='bottom', fill='both')

# Activate the main window.  No code following this executes until 
# the window is closed.
root.mainloop()