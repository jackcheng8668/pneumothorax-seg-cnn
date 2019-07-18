import numpy as np
import os
import sys
import pydicom
import pandas as pd
import tensorflow as tf
keras = tf.keras
from efficientnet_seg.mask_functions import rle2mask
from efficientnet_seg.io.gen_utils import BaseTransformGenerator

class DataGenerator(keras.utils.Sequence):
    """
    Args:
        train_im_path (str): Path to the preprocessed images (.png)
        train_mask_path (str): Path to the preprocessed masks
        augmentations (albumentations transform): either Composed or an individual augmentation
        batch_size (int):
        img_size (int): desired spatial dimensions of the images
        n_channels (int):
        shuffle (bool):
    """
    def __init__(self, train_im_path, train_mask_path,
                 augmentations=None, batch_size=16, img_size=256, n_channels=3, shuffle=True):
        self.batch_size = batch_size
        self.train_im_paths = glob.glob(train_im_path+'/*')

        self.train_im_path = train_im_path
        self.train_mask_path = train_mask_path

        self.img_size = img_size

        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augment = augmentations
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.train_im_paths) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:min((index+1)*self.batch_size,len(self.train_im_paths))]

        # Find list of IDs
        list_IDs_im = [self.train_im_paths[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(list_IDs_im)

        if self.augment is None:
            return X,np.array(y)/255
        else:
            im,mask = [],[]
            for x,y in zip(X,y):
                augmented = self.augment(image=x, mask=y)
                im.append(augmented['image'])
                mask.append(augmented['mask'])
            return np.array(im),np.array(mask)/255

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.train_im_paths))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def data_generation(self, list_IDs_im):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(list_IDs_im),self.img_size,self.img_size, self.n_channels))
        y = np.empty((len(list_IDs_im),self.img_size,self.img_size, 1))

        # Generate data
        for i, im_path in enumerate(list_IDs_im):

            im = np.array(Image.open(im_path))
            mask_path = im_path.replace(self.train_im_path,self.train_mask_path)

            mask = np.array(Image.open(mask_path))


            if len(im.shape)==2:
                im = np.repeat(im[...,None],3,2)

#             # Resize sample
            X[i,] = cv2.resize(im,(self.img_size,self.img_size))

            # Store class
            y[i,] = cv2.resize(mask,(self.img_size,self.img_size))[..., np.newaxis]
            y[y>0] = 255

        return np.uint8(X),np.uint8(y)

class ClassificationGenerator(BaseTransformGenerator):
    """
    Loads data and applies data augmentation with `batchgenerators.transforms`.
    * Supports channels_first
    Attributes:
        fpaths: list of filenames
        batch_size: The number of images you want in a single batch
        transform (Transform instance): If you want to use multiple Transforms, use the Compose Transform.
        step_per_epoch:
        shuffle: boolean
    """
    def __init__(self, fpaths, rle_csv_path, batch_size=2,
                 transform=None, steps_per_epoch=None, shuffle=True):

        BaseTransformGenerator.__init__(self, fpaths=fpaths, batch_size=batch_size, transform=transform,
                                        steps_per_epoch=steps_per_epoch, shuffle=shuffle)
        self.rle_csv_path = rle_csv_path

    def __getitem__(self, idx):
        """
        Defines the fetching and on-the-fly preprocessing of data.
        Args:
            idx: the id assigned to each worker
        Returns:
            (X,Y): a batch of transformed data/labels
            if self.deep_supervision=True, then Y is a list of numpy arrays instead of just a single numpy array
        """
        indexes = self.indexes[idx*self.batch_size:(idx+1)*self.batch_size]
        # Fetches batched IDs for a thread
        fpaths_temp = [self.fpaths[k] for k in indexes]
        X, Y = self.data_gen(fpaths_temp)
        # data augmentation
        if self.transform is not None:
            data_dict = {}
            data_dict["data"] = X
            data_dict = self.transform(**data_dict)
            X = data_dict["data"]
        return (X, Y)

    def data_gen(self, fpaths_temp):
        """
        Generates a batch of data.
        Args:
            fpaths_temp: batched list IDs; usually done by __getitem__
            pos_sample (boolean): if you want to sample an image with a nonzero class or not
        Returns:
            tuple of two lists of numpy arrays: x, y
        """
        images_x = []
        images_y = []
        for fpath in fpaths_temp:
            df = pd.read_csv(self.rle_csv_path, header=None, index_col=0)
            # loads data as a numpy arr and then adds the channel + batch size dimensions
            im = np.array(Image.open(im_path))
            if len(im.shape)==2:
                im = np.repeat(im[...,None],3,2)
            # Resize sample
            x_train = cv2.resize(im,(self.img_size, self.img_size))
            try:
                # Linux support
                encoded_pixels = df.loc[fpath.split('/')[-1][:-4], 1]
            except KeyError:
                # Windows support
                encoded_pixels = df.loc[fpath.split('\\')[-1][:-4], 1]

            y_train = 0 if encoded_pixels == " -1" or encoded_pixels == "-1" else 1
            images_x.append(x_train), images_y.append(y_train)
        images_x, images_y = np.stack(images_x), np.vstack(images_y)
        return (images_x, images_y)
