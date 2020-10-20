import matplotlib.pyplot as plt
import matplotlib.image as mpng
import numpy as np
from scipy import signal
from zplane import zplane

import helper as h

pi = np.pi

"""
For the functions that we apply on the images in the APP
"""


def H_inv(data):
    zeroes = [h.exp_img(0.9, pi / 2), h.exp_img(0.9, -pi / 2), h.exp_img(0.95, pi / 8), h.exp_img(0.95, -pi / 8)]
    poles = [0, -0.99, -0.99, 0.9]  # 2x   -0.99??

    num = np.poly(zeroes)
    denum = np.poly(poles)

    w, Hw = signal.freqz(num, denum)

