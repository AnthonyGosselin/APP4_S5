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
    image.load_image("goldhill_aberrations.npy")
    plt.show() # Necessary to see all plots and images
