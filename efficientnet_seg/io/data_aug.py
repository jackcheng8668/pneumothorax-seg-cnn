import numpy as np
import cv2

from scipy.ndimage.interpolation import rotate
from scipy.ndimage.filters import gaussian_filter

from skimage import exposure

def data_augmentation_all(image, mask=None, p=0.5):
    """
    50% of color inversion, 50% of horizontal flipping and some probability, p, of some other augmentation
    Args:
        image (np.ndarray): of shape (x, y, n_channels)
        mask (np.ndarray or None): same shape as image
    Returns:
        dictionary {"image":..., "mask"...} of the augmented pair. Mask is None if only image is specified.
    """
    if np.random.binomial(1, 0.5):
        image = np.invert(image)
    if np.random.binomial(1, 0.5):
        image = np.fliplr(image)
        if mask is not None:
            mask = np.fliplr(mask)
    if np.random.binomial(1, p):
        image, mask = data_augmentation(image, mask)

    return {"image": image, "mask": mask}

def data_augmentation(image, mask=None):
    """
    Gaussian smoothing, rotation, zooming, or random gamma.
    Args:
        image (np.ndarray): of shape (x, y, n_channels)
        mask (np.ndarray or None): same shape as image
    Returns:
        tuple (image, mask) of the augmented pair. Mask is None if only image is specified.
    """
    # Input should be ONE image with shape: (L, W, CH)
    options = ["gaussian_smooth", "rotate", "zoom", "adjust_gamma"]
    # Probabilities for each augmentation were arbitrarily assigned
    which_option = np.random.choice(options)

    if which_option == "gaussian_smooth":
        sigma = np.random.uniform(0.2, 1.0)
        image = gaussian_filter(image, sigma)
    elif which_option == "zoom":
        # no channel outputs
        image, mask = zoom(image, mask)
    elif which_option == "rotate":
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, reshape=False)
        if mask is not None:
            mask = rotate(mask, angle, reshape=False)
    elif which_option == "adjust_gamma":
        image = image / 255.
        image = exposure.adjust_gamma(image, np.random.uniform(0.75, 1.25))
        image = image * 255.
    # shape checks
    if len(image.shape) == 2: image = np.expand_dims(image, axis=2)
    if mask is not None:
        if len(mask.shape) == 2: mask = np.expand_dims(mask, axis=2)

    return (image, mask)

def zoom(image, mask=None):
    """
    Randomly zooms in the image in the range (0.85*image.shape[0], 0.95*image.shape[0])
    Args:
        image (np.ndarray): of shape (x, y, n_channels)
        mask (np.ndarray or None): same shape as image
    Returns:
        tuple (image, mask) without their channels
    """
    # Assumes image is square
    ## cropping + resizing to zoom
    min_crop = int(image.shape[0]*0.85)
    max_crop = int(image.shape[0]*0.95)
    crop_size = np.random.randint(min_crop, max_crop)
    crop = crop_center(image, crop_size, crop_size)
    if crop.shape[-1] == 1: crop = crop[:,:,0]
    # original used scipy.misc.imresize // based on PIL
    ## Since it's just (cropping, then upsampling), it shouldn't matter which one I use
    # shape shouldn't include the channels dimension
    image = cv2.resize(crop, image.shape[:-1])
    # doing the same with the mask
    if mask is not None:
        crop = crop_center(mask, crop_size, crop_size)
        if crop.shape[-1] == 1: crop = crop[:,:,0]
        # original used scipy.misc.imresize // based on PIL
        ## Since it's just (cropping, then upsampling), it shouldn't matter which one I use
        ## shape shouldn't include the channels dimension
        mask = cv2.resize(crop, mask.shape[:-1], interpolation=cv2.INTER_NEAREST)
    return (image, mask)

def crop_center(img, cropx, cropy):
    y,x = img.shape[:2]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx,:]
