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


def H_inv(data, checkStable=False, printZmap=False):
    # Given TF
    zeroes = [h.exp_img(0.9, pi / 2), h.exp_img(0.9, -pi / 2), h.exp_img(0.95, pi / 8), h.exp_img(0.95, -pi / 8)]
    poles = [0, -0.99, -0.99, 0.9]  # 2x   -0.99??
    num = np.poly(zeroes)
    denum = np.poly(poles)

    # Inverse TF
    # Inverse poles/zeroes and num/denum to have inverse TF
    zeroesInv = poles
    polesInv = zeroes
    numInv = denum
    denumInv = num

    # Verify for pole stability
    if checkStable:
        for pole in polesInv:
            if np.abs(pole) > 1:
                print("Filter Unstable")
            break
            print("Filter Stable")

    # Print zplane for TF and inverse TF
    if printZmap:
        zplane(num, denum)
        zplane(numInv, denumInv)

    w, Hw = signal.freqz(num, denum)
    h.plot(Hw, w, title="Original transfer func")

    w, Hw = signal.freqz(numInv, denumInv)
    h.plot(Hw, w, title="Inverse transfer func")

    dataFiltered = signal.lfilter(numInv, denumInv, data)

    h.imshow(dataFiltered)

    return dataFiltered



def rotate90(data):
    # Rotation matrix
    rot_mat = [[0, -1], [1, 0]]  # [[cos(-90), -sin(-90)], [sin(-90), cos(-90)]]

    print(data.shape)
    x_size, y_size, z_size = data.shape

    x_half = int((x_size-1) / 2)
    y_half = int((y_size-1) / 2)

    data_rotated = np.zeros((y_size, x_size, z_size))
    for y in range(0, y_size):
        for x in range(0, x_size):
            x_centered = x - x_half
            y_centered = y - y_half
            new_centered_pos = np.matmul(rot_mat, np.array([x_centered, y_centered]))

            new_x_ind = new_centered_pos[0] + x_half + 1
            new_y_ind = new_centered_pos[1] + y_half

            data_rotated[new_y_ind][new_x_ind] = data[y][x]

    h.imshow(data_rotated)

    return data_rotated
