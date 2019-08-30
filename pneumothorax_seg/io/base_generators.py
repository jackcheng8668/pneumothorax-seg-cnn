import tensorflow.keras as keras
import numpy as np
import os
from abc import abstractmethod

class BaseGenerator(keras.utils.Sequence):
    """
    Basic framework for generating thread-safe data in keras. (no preprocessing and channels_last)
    Based on https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    Attributes:
      fpaths: filenames (.nii files); must be same for training and labels
      batch_size: int of desired number images per epoch
      shuffle: boolean on whether or not to shuffle the dataset
    """
    def __init__(self, fpaths, batch_size, shuffle=True):
        # lists of paths to images
        self.fpaths = fpaths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.fpaths))

    def __len__(self):
        return int(np.ceil(len(self.fpaths) / float(self.batch_size)))

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.fpaths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        """
        return

    @abstractmethod
    def data_gen(self, fpaths_temp):
        """
        Preprocesses the data
        Args:
            fpaths_temp: temporary batched list of ids (filenames)
        Returns
            x, y
        """
        return
