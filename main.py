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

    img_complete = image.load_image("image_complete.npy", show=True, title="Complete")
    img_aberration = image.load_image("goldhill_aberrations.npy")
    img_rotate = image.load_image("goldhill_rotate.png")
    img_noise = image.load_image("goldhill_bruit.npy", show=False, title="Noisy")
    img_original = image.load_image("goldhill.png", show=True, title="Original")

    img_out_filtered = functions.H_inv(img_aberration if testing else img_complete)
    h.imshow(img_out_filtered, title="After first filter")
    img_out_rotated = functions.rotate90(img_rotate if testing else img_out_filtered, testing)
    img_out_denoised = functions.denoise(img_noise if testing else img_out_rotated)

    plt.show()  # Necessary to see all plots and images