from PIL import Image
import cv2
import numpy as np

def post_process_all(probability_masks, threshold=0.5, min_size=3500):
    """
    Thresholds, and zeroes out small ROIs. Applies to a stack of images instead of per image.
    Args:
        probability_mask (np.ndarray): squeezed output probability mask from a CNN,
            with shape (n_images, h, w)
        threshold (float): threshold value
        min_size (int): minimum number of pixels for an ROI to be not zeroed out.
    Returns:
        predictions (np.ndarray): final thresholded array with small ROIs zeroed out, with
            shape (n_images, 1024, 1024)
    """
    post_processed = [post_process_single(mask, threshold=threshold, min_size=min_size)
                      for mask in probability_masks]
    return np.stack(post_processed)

def post_process_single(probability_mask, threshold=0.5, min_size=3500):
    """
    Thresholds, and zeroes out small ROIs (using the connected components algorithm)
    Edited version of `post_process` in https://www.kaggle.com/rishabhiitbhu/unet-with-resnet34-encoder-pytorch?scriptVersionId=19433645.
    Args:
        probability_mask (np.ndarray): squeezed output probability mask from a CNN,
            with shape (h, w)
        threshold (float): threshold value
        min_size (int): minimum number of pixels for an ROI to be not zeroed out.
    Returns:
        predictions (np.ndarray): final thresholded and transposed mask with small ROIs zeroed out, with
            shape (1024, 1024). Max=255 and dtype=np.uint8.
    """
    h_w = (probability_mask.shape[0], probability_mask.shape[1])
    if h_w != (1024, 1024):
        probability_mask = cv2.resize(probability_mask, (1024, 1024), interpolation=cv2.INTER_LINEAR)
    # thresholding to 0s and 1s
    mask = cv2.threshold(probability_mask, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((1024, 1024), np.float32)
    # iterating through each component and keeping it if it's > min_size
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
    return (predictions.T*255).astype(np.uint8)


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
