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
    testing = True
    verbose = True
    presentation = True

    # Load all images
    img_complete = image.load_image("image_complete.npy", show=False, title="Complete")
    img_aberration = image.load_image("goldhill_aberrations.npy", show=False, title="Aberrations")
    img_rotate = image.load_image("goldhill_rotate.png", show=False, title="Rotate")
    img_noise = image.load_image("goldhill_bruit.npy", show=False, title="Noisy")
    img_original = image.load_image("goldhill.png", show=False, title="Original")

    plt.show() # Not sure why, but there is always an empty graph showing at the beginning

    if not presentation:
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
    else:
        verbose = True

        # Aberration
        targ_img = img_aberration if testing else img_complete
        h.imshow(targ_img, "Original")
        img_out_filtered = functions.H_inv(targ_img, verbose=verbose, in_dB=True)
        plt.show()

        # Rotation
        targ_img = img_rotate if testing else img_out_filtered
        h.imshow(targ_img, "Original")
        img_out_rotated = functions.rotate90(img_rotate if testing else img_out_filtered, testing)
        plt.show()

        # Denoise (1) bilinear transform
        targ_img = img_noise if testing else img_out_rotated
        img_out_denoised_transbi = functions.denoise(targ_img, trans_bi=True, by_hand=False, verbose=verbose, show_plot=False)
        plt.show()

        # Denoise (2) python functions
        targ_img = img_noise if testing else img_out_rotated
        img_out_denoised_pyfunc = functions.denoise(targ_img, trans_bi=False, by_hand=False, verbose=verbose, show_plot=False)
        plt.show()

        # Compare denoise
        h.imshow(targ_img, "Original")
        h.imshow(img_out_denoised_transbi, "Filtre avec transformation bilinéaire")
        h.imshow(img_out_denoised_pyfunc, "Filtre avec fonctions Python")
        plt.show()


        # Image compressions 0.5
        img_out_compressed, passing_matrix = functions.compress_image(img_out_denoised_transbi, compress=True,
                                                                      compression_value=0.5, verbose=False)
        img_out_decompressed, passing_matrix = functions.compress_image(img_out_compressed, compress=False,
                                                                        passing_matrix=passing_matrix, verbose=verbose)

        # Image compression 0.7
        img_out_compressed, passing_matrix = functions.compress_image(img_out_denoised_transbi, compress=True,
                                                                      compression_value=0.7, verbose=False)
        img_out_decompressed, passing_matrix = functions.compress_image(img_out_compressed, compress=False,
                                                                        passing_matrix=passing_matrix, verbose=verbose)
        plt.show()

