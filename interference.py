''' Follows http://physicalmodelingwithpython.blogspot.com/2016/04/make-your-own-gui-with-python.html
Created 3-14-22'''

import tkinter as tk
import matplotlib
# Necessary to get pyplot and tk to play well.
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# Define a bold font:
BOLD = ('Courier', '24', 'bold')

# Create main application window.
root = tk.Tk()

# Create a text box explaining the application.
greeting = tk.Label(text="Create an Interference Pattern", font=BOLD)
greeting.pack(side='top')

# Create a frame for variable names and entry boxes for their values.
frame = tk.Frame(root)
frame.pack(side='top')

# Variables for the calculation, and default values.
amplitudeA = tk.StringVar()
amplitudeA.set('1.0')
frequencyA = tk.StringVar()
frequencyA.set('1.0')

amplitudeB = tk.StringVar()
amplitudeB.set('1.0')
frequencyB = tk.StringVar()
frequencyB.set('1.0')

deltaPhi = tk.StringVar()
deltaPhi.set('0.0')

# Create text boxes and entry boxes for the variables.
# Use grid geometry manager instead of packing the entries in.
row_counter = 0
aa_text = tk.Label(frame, text='Amplitude of 1st wave:') 
aa_text.grid(row=row_counter, column=0)

aa_entry = tk.Entry(frame, width=8, textvariable=amplitudeA)
aa_entry.grid(row=row_counter, column=1)

row_counter += 1
fa_text = tk.Label(frame, text='Frequency of 1st wave:') 
fa_text.grid(row=row_counter, column=0)

fa_entry = tk.Entry(frame, width=8, textvariable=frequencyA)
fa_entry.grid(row=row_counter, column=1)

row_counter += 1
ab_text = tk.Label(frame, text='Amplitude of 2nd wave:') 
ab_text.grid(row=row_counter, column=0)

ab_entry = tk.Entry(frame, width=8, textvariable=amplitudeB)
ab_entry.grid(row=row_counter, column=1)

row_counter += 1
fb_text = tk.Label(frame, text='Frequency of 2nd wave:') 
fb_text.grid(row=row_counter, column=0)

fb_entry = tk.Entry(frame, width=8, textvariable=frequencyB)
fb_entry.grid(row=row_counter, column=1)

row_counter += 1
dp_text = tk.Label(frame, text='Phase Difference:') 
dp_text.grid(row=row_counter, column=0)

dp_entry = tk.Entry(frame, width=8, textvariable=deltaPhi)
dp_entry.grid(row=row_counter, column=1)

# Define a function to create the desired plot.
def make_plot(event=None):
    # Get these variables from outside the function, and update them.
    global amplitudeA, frequencyA, amplitudeB, frequencyB, deltaPhi

    # Convert StringVar data to numerical data.
    aa = float(amplitudeA.get())
    fa = float(frequencyA.get())
    ab = float(amplitudeB.get())
    fb = float(frequencyB.get())
    phi = float(deltaPhi.get())

    # Define the range of the plot.
    t_min = -10
    t_max = 10
    dt = 0.01
    t = np.arange(t_min, t_max+dt, dt)

    # Create the two waves and find the combined intensity.
    waveA = aa * np.cos(fa * t)
    waveB = ab * np.cos(fb * t + phi)
    intensity = (waveA + waveB)**2

    # Create the plot.
    plt.figure()
    plt.plot(t, intensity, lw=3)
    plt.title('Interference Pattern')
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.show()


# Add a button to create the plot.
MakePlot = tk.Button(root, command=make_plot, text="Create Plot")
MakePlot.pack(side='bottom', fill='both')

# Allow pressing <Return> to create plot.
root.bind('<Return>', make_plot)

# Allow pressing <Esc> to close the window.
root.bind('<Escape>', root.destroy)

# Activate the window.
root.mainloop()