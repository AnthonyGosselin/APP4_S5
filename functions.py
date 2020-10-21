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


def H_inv(data, verbose=True, in_dB=True):
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
    if verbose:
        for pole in polesInv:
            if np.abs(pole) > 1:
                print("Filter Unstable")
            break
            print("Filter Stable")

        # Print zplane for TF and inverse TF
        zplane(num, denum, t="H(z) zplane")
        zplane(numInv, denumInv, t="H(z)-1 zplane")

        h.plot_filter(num, denum, t="H(z) (original) transfer function", in_dB=in_dB)
        h.plot_filter(num, denum, t="H(z)-1 (inverse) transfer function", in_dB=in_dB)


    dataFiltered = signal.lfilter(numInv, denumInv, data)
    h.imshow(dataFiltered, t="After H(z)-1 filter")

    return dataFiltered



def rotate90(data, testing=False):
    # Rotation matrix
    rot_mat = [[0, -1], [1, 0]]  # [[cos(-90), -sin(-90)], [sin(-90), cos(-90)]]

    # Complete image doesn't have a third dimension like rotate testing image...
    if testing:
        x_size, y_size, z_size = data.shape
    else:
        x_size, y_size = data.shape

    x_half = int((x_size-1) / 2)
    y_half = int((y_size-1) / 2)

    if testing:
        data_rotated = np.zeros((y_size, x_size, z_size))
    else:
        data_rotated = np.zeros((y_size, x_size))

    for y in range(0, y_size):
        for x in range(0, x_size):
            x_centered = x - x_half
            y_centered = y - y_half
            new_centered_pos = np.matmul(rot_mat, np.array([x_centered, y_centered]))

            new_x_ind = new_centered_pos[0] + x_half + 1
            new_y_ind = new_centered_pos[1] + y_half

            data_rotated[new_y_ind][new_x_ind] = data[y][x]

    h.imshow(data_rotated, t="After 90 degree rotation")

    return data_rotated


def denoise(data, transBi=False, verbose=True):
    fd_pass = 500
    fd_stop = 750
    fe = 1600

    w = 1 / fe

    wd_pass = fd_pass * w
    wd_stop = fd_stop * w

    g_pass = 0.5
    g_stop = 40

    if transBi:

        zeros = [-1, -1]
        poles = [-0.20995, -1]

        num = np.poly(zeros)
        denum = np.poly(poles)

        if verbose:
            zplane(num, denum, t="Cheby order 2 (trans bi) zplane")
            h.plot_filter(num, denum, t="Cheby order 2 (trans bi)", in_dB=False)

        data_denoised = signal.lfilter(num, denum, data)
        h.imshow(data_denoised, t="After Cheby order2 trans bi filter")

    else:

        order = np.zeros(4)
        wn = np.zeros(4)

        # Butterworth
        order[0], wn[0] = signal.buttord(wd_pass, wd_stop, g_pass, g_stop, False, fe)

        # Chebyshev type 1
        order[1], wn[1] = signal.cheb1ord(wd_pass, wd_stop, g_pass, g_stop, False, fe)

        # Chebyshev type 2
        order[2], wn[2] = signal.cheb2ord(wd_pass, wd_stop, g_pass, g_stop, False, fe)

        # Elliptic
        order[3], wn[3] = signal.ellipord(wd_pass, wd_stop, g_pass, g_stop, False, fe)

        lowest_order_index = np.argmin(order)
        print(order)
        print(lowest_order_index)

        if (lowest_order_index == 0):
            print("Butterworth filter order {order}".format(order=order[0]))
            num, denum = signal.butter(order[0], wn[0], 'lowpass', False)
        elif (lowest_order_index == 1):
            print("Cheby1 filter order {order}".format(order=order[1]))
            num, denum = signal.cheby1(order[1], g_pass, wn[1], 'lowpass', False)
        elif (lowest_order_index == 2):
            print("Cheby2 filter order {order}".format(order=order[2]))
            num, denum = signal.cheby2(order[2], g_stop, wn[2], 'lowpass', False)
        else:
            print("Ellip filter order {order}".format(order=order[3]))
            num, denum = signal.ellip(order[3], g_pass, g_stop, wn[3], 'lowpass', False)

        if verbose:
            h.plot_filter(num, denum, t="Filter response", in_dB=True)

        data_denoised = signal.lfilter(num, denum, data)
        h.imshow(data_denoised, "After denoise filter")

    return data_denoised