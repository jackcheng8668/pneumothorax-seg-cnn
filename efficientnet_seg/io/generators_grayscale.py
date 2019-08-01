import numpy as np
import glob
import tensorflow as tf
from efficientnet_seg.io.generators import SegmentationGenerator, ClassificationGenerator
from PIL import Image

class GrayscaleSegmentationGenerator(SegmentationGenerator):
    """
    Generates (image, mask) pairs. Supports `channels_last`
    Args:
        images_dir (str): path to the directory preprocessed images (.png)
        masks_dir (str): path to the masks (.png)
        batch_size (int): Mandatory argument for the desired batch size of the input.
        model_name (str): Either 'densenet', 'inception', or 'xception' to specify the preprocessing
        fpaths (list): of filepaths directly to the training images
        augmentations (albumentations transform): either Composed or an individual augmentation
        shuffle (bool):
    """
    def __init__(self, images_dir, masks_dir, batch_size, model_name=None, fpaths=None, augmentations=None, shuffle=True):
        self.model_name = model_name
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, batch_size=batch_size, fpaths=fpaths, \
                         augmentations=augmentations, shuffle=shuffle)
        self.on_epoch_end()

    def data_gen(self, fpaths_temp):
        """
        Preprocesses the data
        Args:
            fpaths_temp: temporary batched list of ids (filenames)
        Returns
            x, y
        """
        # X : (n_samples, *dim, n_channels)
        # Initialization
        x_batch = []
        y_batch = []
        # Generate data
        for fpath in fpaths_temp:
            # loading the .png images
            x = np.array(Image.open(fpath))[..., np.newaxis]
            mask_path = fpath.replace(self.images_dir, self.masks_dir)
            y = np.array(Image.open(mask_path))[..., np.newaxis]
            if self.model_name is not None:
                x = preprocess_input(x, self.model_name)
            y[y>0] = 255 # does this for data augmentation purposes
            x_batch.append(x), y_batch.append(y)
        X, Y = np.stack(x_batch), np.stack(y_batch)
        return (X, Y)

class GrayscaleClassificationGenerator(ClassificationGenerator):
    """
    Generates (image, classification label). Supports `channels_last`.
    Args:
        images_dir (str): path to the directory preprocessed images (.png)
        masks_dir (str): path to the masks (.png)
        batch_size (int):
        fpaths (list): of filepaths directly to the training images
        augmentations (albumentations transform): either Composed or an individual augmentation
        shuffle (bool):
    """
    def __init__(self, images_dir, masks_dir, batch_size, model_name=None, fpaths=None, augmentations=None, shuffle=True):
        self.model_name = model_name
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, batch_size=batch_size, fpaths=fpaths, \
                         augmentations=augmentations, shuffle=shuffle)
        self.on_epoch_end()

    def data_gen(self, fpaths_temp):
        """
        Generates a batch of data.
        Args:
            fpaths_temp: batched list IDs; usually done by __getitem__
        Returns:
            tuple of numpy arrays: (x, y)
        """
        x_batch = []
        y_batch = []
        for fpath in fpaths_temp:
            # loads data as a numpy arr and then adds the channel + batch size dimensions
            x = np.array(Image.open(fpath))[..., np.newaxis]
            if self.model_name is not None:
                x = preprocess_input(x, self.model_name)
            # creating the classification label from the segmentation mask
            mask_path = fpath.replace(self.images_dir, self.masks_dir)
            y = np.array(Image.open(mask_path))[..., np.newaxis]
            n_unique = np.unique(y).size
            y = 0 if n_unique == 1 else 1

            x_batch.append(x), y_batch.append(y)
        X, Y = np.stack(x_batch), np.vstack(y_batch)
        return (X, Y)

def preprocess_input(x, model_name):
    """
    Preprocess some numpy array input, x, in the style of the user-specified model_name.
    Supports both grayscale and RGB inputs. Assumes channels_last.
    Args:
        x (np.ndarray): (x, y, z, n_channels)
        model_name (str): Either `inception`, `xception`, `mobilenet`, `resnet`, `vgg`, or `densenet`
    """
    x = x.astype("float32")
    if model_name in ("inception","xception","mobilenet"):
        x /= 255.
        x -= 0.5
        x *= 2.
    if model_name in ("densenet"):
        x /= 255.
        if x.shape[-1] == 3:
            x[..., 0] -= 0.485
            x[..., 1] -= 0.456
            x[..., 2] -= 0.406
            x[..., 0] /= 0.229
            x[..., 1] /= 0.224
            x[..., 2] /= 0.225
        elif x.shape[-1] == 1:
            x[..., 0] -= 0.449
            x[..., 0] /= 0.226
    elif model_name in ("resnet","vgg"):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1:
            x[..., 0] -= 115.799
    return x
