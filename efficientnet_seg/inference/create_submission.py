import glob
import cv2
import skimage
import os
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from tqdm import tqdm

"""
To-do:
    make the 2 stages separate functions
    add tqdm for TTA_Classification
    finish segmentation stage

Resources:
    https://www.kaggle.com/iafoss/postprocessing-for-hypercolumns-kernel
    https://github.com/i-pan/kaggle-rsna18/blob/8acd75952c0d6d2edeeadb8f0708d7808c2237a5/src/infer/PredictClassifierEnsemble.py
"""
def create_submission(classification_model, seg_model, batch_size=32, tta=True, classify_csv_fpath=None):
    """
    Performs the cascade. All non-pneumothorax predictions are "-1". All pneumothorax patients
    are then passed to the segmentation model to generate the predicted mask, which is then
    converted to a run-length encoding for the submission file.

    Assuming binary for pneumothorax classification/segmentation.
    """
    test_fpaths = glob.glob('./test/*') # assumes this directory for now
    if classify_csv_fpath is None:
        # Stage 1: Classification predictions
        sub_df = Stage1(classification_model, test_fpaths, batch_size=batch_size, tta=tta)
    else:
        print("Skipping Stage 1...")
        sub_df = pd.read_csv(classify_csv_fpath)
    # Stage 2: Segmentation
    _ = Stage2(seg_model, sub_df, test_fpaths, batch_size=batch_size, tta=tta)

def Stage2(seg_model, sub_df, test_fpaths, batch_size=32, tta=True):
    # Stage 2: Segmentation
    print("Commencing Stage 2: Segmentation of Predicted Pneumothorax (+) Patients")
    # extracting positive only predictions
    test_ids = np.array([Path(fpath).stem for fpath in test_fpaths])
    seg_ids = test_ids[np.where(sub_df["EncodedPixels"] == 1)[0]].tolist()
    img_size = 256 # assuming 256 x 256 for segmentation models as well for now
    x_test = np.asarray([load_input(fpath, img_size) for fpath in test_fpaths if Path(fpath).stem in seg_ids])
    # squeezes are for removing the output classes dimension (1, because binary and sigmoid)
    if tta:
        preds_seg = TTA_Segmentation_All(seg_model, x_test, batch_size=batch_size).squeeze()
    else:
        preds_seg = seg_model.predict(x_test, batch_size=batch_size).squeeze()

    threshold = 0.5 # assuming 0.5 threshold for now.
    preds_seg[preds_seg >= threshold] = 255
    preds_seg[preds_seg < threshold] = 0
    h_w = (preds_seg.shape[1], preds_seg.shape[2])
    if h_w != (1024, 1024):
        print("Resizing the predictions...")
        preds_seg = np.asarray([skimage.transform.resize(pred, (1024, 1024), order=0).T.astype(np.uint8)
                                for pred in preds_seg])
    rles = [mask2rle(pred, 1024, 1024) for pred in preds_seg]
    for id_, rle in zip(seg_ids, rles):
        sub_df.loc[sub_df["ImageId"] == id_, "EncodedPixels"] = rle
    # handling empty masks
    sub_df.loc[sub_df.EncodedPixels=='', 'EncodedPixels'] = '-1'
    sub_df.to_csv('submission_final.csv', index=False)
    print("Stage 2 Completed.")

def Stage1(classification_model, test_fpaths, batch_size=32, tta=True):
    # Stage 1: Classification predictions
    print("Commencing Stage 1: Prediction of Pneumothorax or No Pneumothorax Patients")
    img_size = 256 # assume 256 x 256 images for now
    # Load test set
    x_test = np.asarray([load_input(fpath, img_size) for fpath in test_fpaths])

    ## Hacky fix for binary cases where the output is (N, 1)
    ### Prevents lists being saved as nested lists
    if tta:
        preds_classify = TTA_Classification_All(classification_model, x_test, n_iters=4, batch_size=batch_size).flatten()
    else:
        preds_classify = classification_model.predict(x_test, batch_size=batch_size).flatten()

    test_ids = [Path(fpath).stem for fpath in test_fpaths] # for the df
    threshold = 0.5 # assuming 0.5 threshold for now.
    preds_classify[preds_classify >= threshold] = 1
    preds_classify[preds_classify < threshold] = -1
    # converting elements to integers
    preds_classify = [int(pred) for pred in preds_classify.tolist()]
    # creating first df
    sub_df = pd.DataFrame({'ImageId': test_ids, 'EncodedPixels': preds_classify})
    sub_df.to_csv('submission_classification.csv', index=False)
    print("Stage 1 Completed.")
    return sub_df

def TTA_Classification_All(model, test_arrays, n_iters=4, batch_size=32, seed=88):
    """
    Thin wrapper around for TTA Classification for TTA on all images instead of per image.
    """
    return np.asarray([TTA(model, test_arr, n_iters=n_iters, batch_size=batch_size, seed=seed)
                       for test_arr in test_arrays])

def TTA_Classification(model, test_array, n_iter=4, batch_size=32, seed=88):
    """
    Test-time augmentation with array inversion, horizontal and vertical flipping,
    gaussian smoothing, random rotations, and random zooms.
    NOTE: THIS IS PER-IMAGE BECAUSE OF THE MEMORY LIMITATIONS.
    Args:
        model (instance of keras.models.Model): should predict an array with shape (N, n_classes)
        test_array (np.ndarray): shape of (H, W, C)
        batch_size: the batch size for prediction
    Returns:
        preds_test (np.ndarray): predicted probability for test_array (n_classes,)
    """
    # for reproducibility
    np.random.seed(seed)
    # creating arrays to store the augmented images
    original_img_array = np.empty((niter+1, test_array.shape[0], test_array.shape[1], test_array.shape[2]))
    inverted_img_array = original_img_array.copy()
    hflipped_img_array = original_img_array.copy()
    # 1st round of augmentations: flipping and inversion
    original_img = test_array.copy()
    inverted_img = np.invert(test_array.copy())
    hflipped_img = np.fliplr(test_array.copy())
    original_img_array[0] = original_img
    inverted_img_array[0] = inverted_img
    hflipped_img_array[0] = hflipped_img

    # data augmentation: vertical flipping, gaussian smoothing, random rotations, and random zooms.
    for each_iter in range(niter):
        original_img_array[each_iter+1] = data_augmentation(original_img)
        inverted_img_array[each_iter+1] = data_augmentation(inverted_img)
        hflipped_img_array[each_iter+1] = data_augmentation(hflipped_img)
    tmp_array = np.vstack((original_img_array, inverted_img_array, hflipped_img_array))
    # preprocessing // for now, no preprocessing
    # tmp_array = preprocess_input(tmp_array, model_name)

    n_classes = int(model.get_output_at(-1).get_shape()[1])
    prediction = np.mean(model.predict(tmp_array), axis=0)
    return prediction

def TTA_Segmentation_All(model, test_arrays, batch_size=32):
    """
    Test-time augmentation with only left-right flipping for segmentation models.
    Also predicts for the original images as well.
    Args:
        model (instance of keras.models.Model): should predict a mask that has the same (N, H, W) as test_arrays.
        test_arrays (np.ndarray): shape of (N, H, W, C)
        batch_size: the batch size for prediction
    Returns:
        preds_test (np.ndarray): averaged predicted activation map with shape of (N, H, W, n_classes)
    """
    # predicting original images
    preds_test = model.predict(test_arrays, batch_size=batch_size)
    # predicting the flipped versions
    x_test = np.asarray([np.fliplr(x) for x in test_arrays])
    preds_test_tta = model.predict(x_test, batch_size=batch_size)
    # flipping them back to their original state.
    preds_test_tta = np.asarray([np.fliplr(x) for x in preds_test_tta])
    preds_test = np.mean([preds_test, preds_test_tta], axis=0)
    return preds_test

def data_augmentation(image):
    # Input should be ONE image with shape: (L, W, CH)
    options = ["gaussian_smooth", "vertical_flip", "rotate", "zoom", "adjust_gamma"]
    # Probabilities for each augmentation were arbitrarily assigned
    which_option = np.random.choice(options)
    if which_option == "vertical_flip":
        image = np.fliplr(image)

    if which_option == "gaussian_smooth":
        sigma = np.random.uniform(0.2, 1.0)
        image = gaussian_filter(image, sigma)
    elif which_option == "zoom":
      # Assumes image is square
        min_crop = int(image.shape[0]*0.85)
        max_crop = int(image.shape[0]*0.95)
        crop_size = np.random.randint(min_crop, max_crop)
        crop = crop_center(image, crop_size, crop_size)
        if crop.shape[-1] == 1: crop = crop[:,:,0]
        image = scipy.misc.imresize(crop, image.shape)
    elif which_option == "rotate":
        angle = np.random.uniform(-15, 15)
        image = rotate(image, angle, reshape=False)
    elif which_option == "adjust_gamma":
        image = image / 255.
        image = exposure.adjust_gamma(image, np.random.uniform(0.75, 1.25))
        image = image * 255.
    if len(image.shape) == 2: image = np.expand_dims(image, axis=2)
    return image

def preprocess_input(x, model):
    x = x.astype("float32")
    if model in ("inception","xception","mobilenet"):
        x /= 255.
        x -= 0.5
        x *= 2.
    if model in ("densenet"):
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
    elif model in ("resnet","vgg"):
        if x.shape[-1] == 3:
            x[..., 0] -= 103.939
            x[..., 1] -= 116.779
            x[..., 2] -= 123.680
        elif x.shape[-1] == 1:
            x[..., 0] -= 115.799
    return x

def load_input(fpath, img_size=256):
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
    return np.repeat(arr[...,None], 3, 2)

import numpy as np

def mask2rle(img, width, height):
    rle = []
    lastColor = 0;
    currentPixel = 0;
    runStart = -1;
    runLength = 0;

    for x in range(width):
        for y in range(height):
            currentColor = img[x][y]
            if currentColor != lastColor:
                if currentColor == 255:
                    runStart = currentPixel;
                    runLength = 1;
                else:
                    rle.append(str(runStart));
                    rle.append(str(runLength));
                    runStart = -1;
                    runLength = 0;
                    currentPixel = 0;
            elif runStart > -1:
                runLength += 1
            lastColor = currentColor;
            currentPixel+=1;

    return " ".join(rle)

def rle2mask(rle, width, height):
    mask= np.zeros(width* height)
    array = np.asarray([int(x) for x in rle.split()])
    starts = array[0::2]
    lengths = array[1::2]

    current_position = 0
    for index, start in enumerate(starts):
        current_position += start
        mask[current_position:current_position+lengths[index]] = 255
        current_position += lengths[index]

    return mask.reshape(width, height)
