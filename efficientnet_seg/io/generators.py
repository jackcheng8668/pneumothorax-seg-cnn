import numpy as np
import glob
import pandas as pd
import tensorflow as tf
keras = tf.keras
from efficientnet_seg.io.base_generators import BaseGenerator
from PIL import Image

class SegmentationGenerator(BaseGenerator):
    """
    Generates (image, mask) pairs. Supports `channels_last`
    Args:
        images_dir (str): path to the directory preprocessed images (.png)
        masks_dir (str): path to the directory of masks (.png)
        augmentations (albumentations transform): either Composed or an individual augmentation
        batch_size (int):
        shuffle (bool):
    """
    def __init__(self, images_dir, masks_dir, batch_size, augmentations=None, shuffle=True):
        fpaths = glob.glob(images_dir+'/*')
        super().__init__(fpaths=fpaths, batch_size=batch_size, shuffle=shuffle)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augmentations
        self.on_epoch_end()

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size: min((index+1)*self.batch_size, len(self.fpaths))]

        # Find list of IDs
        fpaths_temp = [self.fpaths[k] for k in indexes]

        # Generate data
        X, Y = self.data_gen(fpaths_temp)

        if self.augment is None:
            return X, np.array(Y)/255
        else:
            im,mask = [], []
            for x,y in zip(X,Y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im), np.array(mask)/255

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
            x = np.array(Image.open(fpath))
            mask_path = fpath.replace(self.images_dir, self.masks_dir)
            y = np.array(Image.open(mask_path))[..., np.newaxis]
            # Adjusting the shape of x if need be
            if len(x.shape)==2:
                x = np.repeat(x[...,None], 3, 2)
            y[y>0] = 255
            x_batch.append(x), y_batch.append(y)
        X, Y = np.stack(x_batch), np.stack(y_batch)
        return (X, Y)

class ClassificationGenerator(BaseGenerator):
    """
    Generates (image, classification label). Supports `channels_last`.
    Args:
        images_dir (str): path to the directory preprocessed images (.png)
        augmentations (albumentations transform): either Composed or an individual augmentation
        batch_size (int):
        shuffle (bool):
    """
    def __init__(self, images_dir, masks_dir, batch_size, augmentations=None, shuffle=True):
        fpaths = glob.glob(images_dir+'/*')
        super().__init__(fpaths=fpaths, batch_size=batch_size, shuffle=shuffle)
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.augment = augmentations
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
            return (X, Y)
        else:
            X = np.stack([self.augment(image=x)['image'] for x in X])
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
            x = np.array(Image.open(fpath))
            if len(x.shape)==2:
                x = np.repeat(x[..., None], 3, 2)
            # creating the label
            mask_path = fpath.replace(self.images_dir, self.masks_dir)
            y = np.array(Image.open(mask_path))[..., np.newaxis]
            n_unique = np.unique(y).size
            y = 0 if n_unique == 1 else 1

            x_batch.append(x), y_batch.append(y)
        X, Y = np.stack(x_batch), np.vstack(y_batch)
        return (X, Y)
