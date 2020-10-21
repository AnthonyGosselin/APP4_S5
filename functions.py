import matplotlib.pyplot as plt
import matplotlib.image as mpng
import numpy as np
from scipy import signal
from zplane import zplane
import sympy as sp

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

    h.imshow(data_rotated)

    return data_rotated


def denoise(data, transBi=False):
    fd_pass = 500
    fd_stop = 750
    fe = 1600

    w = 1 / fe

    wd_pass = fd_pass * w
    wd_stop = fd_stop * w

    g_pass = 0.5
    g_stop = 40

    if transBi:

        # Gauchissement
        wa_pass = h.gauchissement(fd_pass, fe)
        #wa_stop = h.gauchissement(fd_stop, fe)

        # Write H(s) -> H(z) function
        z = sp.Symbol('z')
        s = 2 * fe * (z-1) / (z+1)
        H = 1 / ((s/wa_pass)**2 + np.sqrt(2)*(s/wa_pass) + 1)
        H = sp.simplify(H)
        print(H)

        # Seperate num and denum into fractions
        num, denum = sp.fraction(H)
        # Put them in polynomial form
        num = sp.poly(num)
        denum = sp.poly(denum)

        # Find zeros and poles
        zeros = sp.roots(num)
        poles = sp.roots(denum)
        print("Zeros and poles: " + str(zeros) + ", " + str(poles))

        # Extract all coefficients and write it in np.array form
        num = np.float64(np.array(num.all_coeffs()))
        denum = np.float64(np.array(denum.all_coeffs()))
        print("Num and Denum: " + str(num, ) + ", " + str(denum))
        zplane(num, denum)

        data_denoised = signal.lfilter(num, denum, data)
        h.imshow(data_denoised, "After bilinear noise filter")


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
            num, denum = signal.butter(order[0], wn[0], 'lowpass', False)
        elif (lowest_order_index == 1):
            num, denum = signal.cheby1(order[1], g_pass, wn[1], 'lowpass', False)
        elif (lowest_order_index == 2):
            num, denum = signal.cheby2(order[2], g_stop, wn[2], 'lowpass', False)
        else:
            num, denum = signal.ellip(order[3], g_pass, g_stop, wn[3], 'lowpass', False)


        h.plot_filter(num, denum, title="Filter response", in_dB=True)

        data_denoised = signal.lfilter(num, denum, data)
        h.imshow(data_denoised, "After python function noise filter")

    return data_denoised