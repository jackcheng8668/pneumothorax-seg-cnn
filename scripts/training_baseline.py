import tensorflow.keras.backend as K
import tensorflow as tf
import numpy as np
import cv2
import shutil
import glob
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from albumentations import (
    Compose, HorizontalFlip, CLAHE, HueSaturationValue,
    RandomBrightness, RandomContrast, RandomGamma,OneOf,
    ToFloat, ShiftScaleRotate,GridDistortion, ElasticTransform, JpegCompression, HueSaturationValue,
    RGBShift, RandomBrightness, RandomContrast, Blur, MotionBlur, MedianBlur, GaussNoise, CenterCrop,
    IAAAdditiveGaussianNoise, GaussNoise, OpticalDistortion
)
from efficient_seg.models.uefficientnet import UEfficientNet
from efficient_seg.training.callbacks import *
from efficient_seg.models.losses_metrics import bce_dice_loss, my_iou_metric
from efficientnet_seg.io.generators import DataGenerator

def get_transforms():
    """
    Quick utility function to return the augmentations for the training/validation generators
    """
    aug_train = Compose([
        HorizontalFlip(p=0.5),
        OneOf([
            RandomContrast(),
            RandomGamma(),
            RandomBrightness(),
             ], p=0.3),
        OneOf([
            ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            GridDistortion(),
            OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
        ToFloat(max_value=1)
    ],p=1)

    aug_val = Compose([
        ToFloat(max_value=1)
    ],p=1)
    return aug_train, aug_val

def move_to_dir(directory, base_fns):
    """
    Moves files to a new directory
    """
    os.mkdir(directory)
    for full_fn in base_fns:
        fn = Path(full_fn).name
        shutil.move(full_fn, os.path.join(directory, fn))

if __name__ == "__main__":
    # after unzipping
    # Creating train/validation sets
    all_train_fn = glob.glob("/content/train/*")
    total_samples = len(all_train_fn)
    idx = np.arange(total_samples)
    train_fn, val_fn = train_test_split(all_train_fn, stratify=mask_df.labels, test_size=0.1, random_state=10)

    print("No. of train files: {0}".format(len(train_fn)))
    print("No. of val files: {0}".format(len(val_fn)))

    masks_train_fn = [fn.replace("/content/train", "/content/masks") for fn in train_fn]
    masks_val_fn = [fn.replace("/content/train", "/content/masks") for fn in val_fn]
    train_im_path, train_mask_path = "/content/keras_im_train", "/content/keras_mask_train"
    val_im_path, val_mask_path = "/content/keras_im_val", "/content/keras_mask_val"
    dirs = [train_im_path, train_mask_path, val_im_path, val_mask_path]
    fns = [train_fn, masks_train_fn, val_fn, masks_val_fn]
    for directory, fn_list in zip(dirs, fns):
        move_to_dir(directory, fn_list)

    h,w,batch_size = 256, 256, 16
    K.clear_session()
    img_size = 256
    model = UEfficientNet(input_shape=(img_size, img_size, 3), dropout_rate=0.25)
    model.compile(loss=bce_dice_loss, optimizer="adam", metrics=[my_iou_metric])

    epochs = 70
    snapshot = SnapshotCallbackBuilder(nb_epochs=epochs, nb_snapshots=1, init_lr=1e-3)
    batch_size = 16
    swa = SWA("/content/keras_swa.model", 67)
    # Generators
    aug_train, aug_val = get_transforms()
    training_generator = DataGenerator(augmentations=aug_train, img_size=img_size)
    validation_generator = DataGenerator(train_im_path=valid_im_path,
                                         train_mask_path=valid_mask_path, augmentations=aug_val,
                                         img_size=img_size)

    history = model.fit_generator(generator=training_generator,
                                validation_data=validation_generator,
                                use_multiprocessing=False,
                                epochs=epochs, verbose=2,
                                callbacks=snapshot.get_callbacks())
