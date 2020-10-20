import matplotlib.pyplot as plt
import matplotlib.image as mpng
import numpy as np

"""
For basic image manipulation functions
"""


# To load image data and display it
def load_image(image_name, show=True, gray=True):
    if gray:
        plt.gray()

    image_path = './images_in/'+image_name
    if image_name.find(".npy") != -1:
        img = np.load(image_path)
    elif image_name.find(".png") != -1:
        img = mpng.imread(image_path)
    else:
        raise Exception("File name must include either .png or .npy file extension")

    if show:
        plt.imshow(img)

    return img
