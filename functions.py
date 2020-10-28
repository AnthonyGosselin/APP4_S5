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


def H_inv(data, verbose=True, in_dB=True):
    # Given TF
    zeroes = [h.exp_img(0.9, pi / 2), h.exp_img(0.9, -pi / 2),
              h.exp_img(0.95, pi / 8), h.exp_img(0.95, -pi / 8)]

    poles = [0, -0.99, -0.99, 0.9]  # 2x   -0.99??
    num = np.poly(zeroes)
    denum = np.poly(poles)

    # Inverse TF
    # Inverse poles/zeroes and num/denum to have inverse TF
    zeroes_inv = poles
    poles_inv = zeroes
    num_inv = denum
    denum_inv = num


    if verbose:
        # Verify for pole stability
        pole_stable = True
        for pole in poles_inv:
            if np.abs(pole) > 1:
                pole_stable = False
                break
        if pole_stable:
            print("Filter Stable")
        else:
            print("Filter Unstable")

        # Print zplane for TF and inverse TF
        # zplane(num, denum, t="H(z) zplane")
        zplane(num_inv, denum_inv, t="H(z)-1 zplane")

        # h.plot_filter(num, denum, t="H(z) (original) transfer function", in_dB=in_dB)
        # h.plot_filter(num_inv, denum_inv, t="H(z)-1 (inverse) transfer function", in_dB=in_dB)

    data_filtered = signal.lfilter(num_inv, denum_inv, data)
    h.imshow(data_filtered, t="After H(z)-1 filter")

    return data_filtered



def rotate90(data, testing=False):

    # Rotation matrix
    rot_mat = [[0, -1],  # [[cos(-90), -sin(-90)],
               [1, 0]]   # [ sin(-90), cos(-90)]]

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
            # Compute coordinates for centered image
            x_centered = x - x_half
            y_centered = y - y_half

            # Rotate at origin
            new_centered_pos = np.matmul(rot_mat,
                               np.array([x_centered, y_centered]))

            # Translate back to position
            new_x_ind = new_centered_pos[0] + x_half + 1
            new_y_ind = new_centered_pos[1] + y_half
            data_rotated[new_y_ind][new_x_ind] = data[y][x]

    h.imshow(data_rotated, t="After 90 degree rotation")

    return data_rotated


def denoise(data, trans_bi=False, by_hand=False, verbose=True, show_plot=True):
    fd_pass = 500
    fd_stop = 750
    fe = 1600

    w = 1 / fe

    wd_pass = fd_pass
    wd_stop = fd_stop

    g_pass = 0.5
    g_stop = 40

    if trans_bi:
        if not by_hand:
            # "Gauchissement"
            wa_pass = h.gauchissement(fd_pass, fe)
            if verbose: print(wa_pass)

            # Write H(s) -> H(z) function
            z = sp.Symbol('z')
            s = 2 * fe * (z-1) / (z+1)
            H = 1 / ((s/wa_pass)**2 + np.sqrt(2)*(s/wa_pass) + 1)
            H = sp.simplify(H)
            if verbose: print(H)

            # Seperate num and denum into fractions
            num, denum = sp.fraction(H)

            # Put them in polynomial form
            num = sp.poly(num)
            denum = sp.poly(denum)

            # Extract all coefficients and write it in np.array form
            k = 1 / 2.3914
            num = np.float64(np.array(num.all_coeffs())) * k
            denum = np.float64(np.array(denum.all_coeffs())) * k
            if verbose:
                print("Num and Denum: " + str(num) + ", " + str(denum))

            # Extract zeros and poles by finding roots of num and den
            zeros = np.roots(num)
            poles = np.roots(denum)
            if verbose:
                print("Zeros and poles: " + str(zeros) + ", " + str(poles))

            if verbose:
                zplane(num, denum, t="zPlane 2nd order butterworth bilinéaire filter")
                h.plot_filter(num, denum, t="2nd order butterworth bilinéaire filter", in_dB=True, in_freq=True, fe=fe)

        else:
            # Done by hand
            zeros = [-1, -1]
            poles = [np.complex(-0.2314, 0.3951), np.complex(-0.2314, -0.3951)]
            k = 1 / 2.39

            num = np.poly(zeros) * k
            num = k * np.poly(zeros)
            denum = np.poly(poles)

            if verbose:
                print("Num and Denum: " + str(num, ) + ", " + str(denum))
                zplane(num, denum, t="Butterworth order 2 (trans. bilinéaire) zplane")
                h.plot_filter(num, denum, t="Butterworth order 2 (trans. bilinéaire)", in_dB=True, in_freq=True, fe=fe)

        data_denoised = signal.lfilter(num, denum, data)

        if show_plot:
            h.imshow(data_denoised, t="After Butterworth order 2 trans. bilinéaire filter")

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
        if verbose:
            print(order)
            print(lowest_order_index)
            print(wn)

        if (lowest_order_index == 0):
            filter_name = "Butterworth filter order {order}".format(order=order[0])
            num, denum = signal.butter(order[0], wn[0], 'lowpass', False, 'ba', fe)
        elif (lowest_order_index == 1):
            filter_name = "Cheby1 filter order {order}".format(order=order[1])
            num, denum = signal.cheby1(order[1], g_pass, wn[1], 'lowpass', False, 'ba', fe)
        elif (lowest_order_index == 2):
            filter_name = "Cheby2 filter order {order}".format(order=order[2])
            num, denum = signal.cheby2(order[2], g_stop, wn[2], 'lowpass', False, 'ba', fe)
            filter_name = "Cheby2 " + str(order[2]) + " order"
        else:
            filter_name = "Ellip filter order {order}".format(order=order[3])
            num, denum = signal.ellip(order[3], g_pass, g_stop, wn[3], 'lowpass', False, 'ba', fe)


        if verbose:
            print(filter_name)
            filter_response_str = "Filter response " + filter_name
            zplane_str = "zPlane " + filter_name
            h.plot_filter(num, denum, t=filter_response_str, in_dB=True, in_freq=True, fe=fe)
            zplane(num, denum, t=zplane_str)

        data_denoised = signal.lfilter(num, denum, data)

        if show_plot:
            h.imshow(data_denoised, "After python function noise filter")

    return data_denoised


def compress_image(data, compress=True, compression_value=0.5, passing_matrix=None, verbose=False):

    if compress:
        # Find covariance matrix and then eigenvalues and eigenvectors
        cov_matrix = np.cov(data, rowvar=True)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        passing_matrix = np.transpose(eigenvectors)
    else:
        # Since original base is orthogonal, inverse = transpose
        passing_matrix = np.transpose(passing_matrix)
        zero_appending_matrix = np.zeros((int(passing_matrix.shape[0]-data.shape[0]), data.shape[1]))
        data = np.append(data, zero_appending_matrix, axis=0)

        # Find 0 ratio that was used for compressing image
        compression_value = float(zero_appending_matrix.shape[0]/passing_matrix.shape[0])

    data_compressed = np.matmul(passing_matrix, data)

    # Only send values of the matrix that do not have zeros
    if compress:
        new_compressed_image = data_compressed[0:int((1-compression_value)*data_compressed.shape[0])]
        data_compressed = new_compressed_image

    if verbose:
        if compress:
            name = "Compressed image with %.1f compression ratio" % compression_value
            h.imshow(data_compressed, t=name)
        else:
            name = "Compressed image with %.1f compression ratio" % compression_value
            h.imshow(data_compressed, t=name)

    return data_compressed, passing_matrix
