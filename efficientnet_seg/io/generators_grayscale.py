import numpy as np
import glob
import tensorflow as tf
import albumentations

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
            * Note: This can also just be a function; must take in the `image` and `mask` arguments and
            return a dictionary with the keys: `image`, `mask`. An example is `io.data_aug.data_augmentation_all`.
        shuffle (bool):
    """
    def __init__(self, images_dir, masks_dir, batch_size, model_name=None, fpaths=None, augmentations=None, shuffle=True):
        self.model_name = model_name
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, batch_size=batch_size, fpaths=fpaths, \
                         augmentations=augmentations, shuffle=shuffle)
        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size: min((index+1)*self.batch_size, len(self.fpaths))]

        # Find list of IDs
        fpaths_temp = [self.fpaths[k] for k in indexes]

        # Generate data
        X, Y = self.data_gen(fpaths_temp)

        # only preprocesses the input when there is no data augmentation
        if self.augment is None:
            if self.model_name is not None:
                X = preprocess_input(X, self.model_name)
            return X, np.array(Y)/255
        else:
            # Augmentation
            im, masks = [], []
            for x,y in zip(X,Y):
                augmented = self.augment(image=x, mask=y)
                # The output of a self.augment should be a dictionary {"image":..., "mask":...}
                im.append(augmented['image']), masks.append(augmented['mask'])
            X, Y = np.asarray(im), np.asarray(masks)/255
            # preprocessing
            if self.model_name is not None:
                X = preprocess_input(X, self.model_name)
            return X, Y

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
            * Note: This can also just be a function; must take in the `image` and `mask` arguments and
            return a dictionary with the keys: `image`, `mask`. An example is `io.data_aug.data_augmentation_all`.
        shuffle (bool):
    """
    def __init__(self, images_dir, masks_dir, batch_size, model_name=None, fpaths=None, augmentations=None, shuffle=True):
        self.model_name = model_name
        super().__init__(images_dir=images_dir, masks_dir=masks_dir, batch_size=batch_size, fpaths=fpaths, \
                         augmentations=augmentations, shuffle=shuffle)
        self.on_epoch_end()

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Args:
            idx: the id assigned to each worker
        Returns:
            (X,Y): a batch of transformed data/labels
        """
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        fpaths_temp = [self.fpaths[k] for k in indexes]
        X, Y = self.data_gen(fpaths_temp)
        # data augmentation
        if self.augment is None:
            if self.model_name is not None:
                X = preprocess_input(X, self.model_name)
            return (X, Y)
        else:
            # The output of a self.augment should be a dictionary {"image":..., "mask":...}
            X = np.stack([self.augment(image=x)['image'] for x in X])
            if self.model_name is not None:
                X = preprocess_input(X, self.model_name)
            return (X, Y)

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
