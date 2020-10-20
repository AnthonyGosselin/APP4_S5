import numpy as np
import matplotlib.pyplot as plt

"""
For little helper functions
"""

pi = np.pi
e = np.e


def dB(x):
    return 20*np.log10(x)


def plot(y, x, plot_x=True, title=""):
    plt.figure()
    if plot_x:
      plt.plot(x, y)
    else:
      plt.plot(y)

    plt.title(title)

def imshow(data):
    plt.figure()
    plt.imshow(data)

def exp_img(mod, ang):
    return mod*e**(np.complex(0, ang))
