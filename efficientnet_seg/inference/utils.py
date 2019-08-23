from PIL import Image
import cv2
import numpy as np

def load_input(fpath, img_size=256, channels=3):
    """
    Loads and resizes a .png file.

    Args:
        fpath (str): file path to a .png file to load
        img_size (int): representing the height and width of the image to resize to
    Returns:
        arr (np.ndarray): shape (img_size, img_size, channels)
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

def batch_test_fpaths(test_fpaths, batch_size=320):
    """
    Batch the test filepaths into a list of batched sublists of filepaths.
    Args:
        test_fpaths (list): of filepaths to the test images
        batch_size (int): number of images in each sublist
    Returns:
        a list of lists of filepaths
    """
    n_splits = len(test_fpaths) // batch_size
    def chunks(l, n):
        """
        https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
        For evenly splitting some list, l, into evenly sized chunks with size, n.
        Returns:
            generator that generates batched lists
        """
        n = max(1, n)
        return (l[i:i+n] for i in range(0, len(l), n))
    chunks_gen = chunks(test_fpaths, batch_size)
    test_fpaths = [sublist for sublist in chunks_gen]
    return test_fpaths
