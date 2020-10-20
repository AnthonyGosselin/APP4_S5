import numpy as np
import matplotlib.pyplot as plt

"""
For little helper functions
"""

pi = np.pi
e = np.e


def dB(x):
    return 20*np.log10(x)


def plot(y, x=None, title=""):
    plt.figure()
    if x:
      plt.plot(x, y)
    else:
      plt.plot(y)

    plt.title(title)


def exp_img(mod, ang):
    return mod*e**(np.complex(0, ang))
