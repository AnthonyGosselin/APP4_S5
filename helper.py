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
    return wa

def plot_filter(num, denum, t="", in_dB=False, norm=True):
    w, Hw = signal.freqz(num, denum)

    if in_freq:
        w = w/(2*pi) * fe*2

    mod = dB(np.abs(Hw)) if in_dB else np.abs(Hw)
    plot(mod, w, t=t)
    plt.ylabel("Amplitude (dB)") if in_dB else plt.ylabel("Amplitude")
    plt.xlabel("Fréquence normalisée (rad/échantillon)") if norm else plt.xlabel("Fréquence (Hz)")

    plt.ylabel("Amplitude (dB)") if in_dB else plt.ylabel("Amplitude")
    plt.xlabel("Fréquence normalisée (rad/échantillon)") if not in_freq else plt.xlabel("Fréquence (Hz)")
