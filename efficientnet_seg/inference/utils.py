from PIL import Image
import cv2
import numpy as np
def load_input(fpath, img_size=256, channels=3):
    """
    Loads and resizes a .png file.
    Args:
        fpath (str): file path to a .png file to load
        img_size (int): representing the height and width of the image to resize to
    """
    arr = np.array(Image.open(fpath))
    resize_shape = (img_size, img_size)
    # prevents unnecesssary resizing
    if arr.shape != resize_shape:
        arr = cv2.resize(arr, resize_shape)
    # repeating for RGB inputs
    if channels == 3:
        arr = np.repeat(arr[..., None], 3, 2)
    elif channels == 1:
        arr = arr[..., np.newaxis]
    else:
        raise Exception("The models in this repository only support grayscale or RGB inputs.")
    return arr
