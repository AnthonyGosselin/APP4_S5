import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

"""
For little helper functions
"""

pi = np.pi
e = np.e


def dB(x):
    return 20*np.log10(x)


def plot(y, x, plot_x=True, t=""):
    plt.figure()
    if plot_x:
      plt.plot(x, y)
    else:
      plt.plot(y)

    plt.title(t)

def imshow(data, t=""):
    plt.figure()
    plt.imshow(data)
    plt.title(t)

def exp_img(mod, ang):
    return mod*e**(np.complex(0, ang))


def gauchissement(fd, fe):
    wd = 2 * np.pi * fd / fe
    wa = 2 * fe * np.tan(wd / 2)
    fa = wa / (2 * np.pi)
    return fa

def plot_filter(num, denum, t="", in_dB=False):
    w, Hw = signal.freqz(num, denum)

    mod = dB(np.abs(Hw)) if in_dB else np.abs(Hw)
    plot(mod, w, t=t)
