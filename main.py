import matplotlib.pyplot as plt
import matplotlib.image as mpng
import numpy as np
from scipy import signal
from zplane import zplane

import image
import functions
import helper as h

pi = np.pi


"""
Mettons les fonctions dans des fichiers externes, ici on aura juste du code pour appeler les fonctions
"""

if __name__ == "__main__":
    # If testing = false, use complete image, else use individual problem images
    testing = False
    verbose = True

    # Load all images
    img_complete = image.load_image("image_complete.npy", show=False, title="Complete")
    img_aberration = image.load_image("goldhill_aberrations.npy", show=False, title="Aberrations")
    img_rotate = image.load_image("goldhill_rotate.png", show=False, title="Rotate")
    img_noise = image.load_image("goldhill_bruit.npy", show=False, title="Noisy")
    img_original = image.load_image("goldhill.png", show=False, title="Original")

    # Use functions with the right images, goldhill if testing each function individually and complete if final (same image recursively for all)
    img_out_filtered = functions.H_inv(img_aberration if testing else img_complete, verbose=verbose, in_dB=True)
    img_out_rotated = functions.rotate90(img_rotate if testing else img_out_filtered, testing)
    img_out_denoised = functions.denoise(img_noise if testing else img_out_rotated, trans_bi=True, by_hand=False, verbose=verbose)

    # Image compressions (first 0.5, then 0.7)
    img_out_compressed, passing_matrix = functions.compress_image(img_out_denoised, compress=True, compression_value=0.5, verbose=verbose)
    img_out_decompressed, passing_matrix = functions.compress_image(img_out_compressed, compress=False, passing_matrix=passing_matrix, verbose=verbose)
    img_out_compressed, passing_matrix = functions.compress_image(img_out_denoised, compress=True, compression_value=0.7, verbose=verbose)
    img_out_decompressed, passing_matrix = functions.compress_image(img_out_compressed, compress=False, passing_matrix=passing_matrix, verbose=verbose)

    plt.show()  # Necessary to see all plots and images