import os
import numpy as np
import pandas as pd

from pathlib import Path
from efficientnet_seg.io.data_aug import data_augmentation
from efficientnet_seg.inference.utils import load_input
from efficientnet_seg.io.generators_grayscale import preprocess_input

def Stage1(classification_model, test_fpaths, channels=3, img_size=256, batch_size=32, tta=True,
           threshold=0.5, model_name=None):
    """
    For the first (classification) stage of the classification/segmentation cascade. It assumes that the
    classification_model was trained on the regular dataset.

    Args:
        classification_model (a single tf.keras.model.Model or keras.model.Model or a list of them): assumes
            that they all need the same input. When `classification_model` is a list/tuple, the models are
            ensembled (predictions are averaged.)
        test_fpaths (list or tuple): of file paths to the test images
        channels (int): The number of input channels. Defaults to 3.
        img_size (int): The size of each square input image. Defaults to 256.
        batch_size (int): model prediction batch size
        tta (boolean): whether or not to apply test-time augmentation.
        threshold (float): threshold for the predicted arrays where
            anything >= threshold = 255 and anything < threshold = 0. Defaults to 0.5.
        model_name (str): Either 'densenet', 'inception', or 'xception' to specify the preprocessing method
    Returns:
        sub_df (pd.DataFrame): the classification submission data frame (Encoded pixels are 1/-1 for pneumothorax/no pneumothorax).
    """
    # Stage 1: Classification predictions
    print("Commencing Stage 1: Prediction of Pneumothorax or No Pneumothorax Patients")
    # Load test set
    x_test = np.asarray([load_input(fpath, img_size, channels=channels) for fpath in test_fpaths])
    if model_name is not None:
        x_test = preprocess_input(x_test, model_name)
    ## Hacky fix for binary cases where the output is (N, 1)
    ### Prevents lists being saved as nested lists
    if tta:
        # ensembling with TTA
        if isinstance(classification, (list, tuple)):
            # stacking across the batch_size dimension
            preds_classify = np.mean(np.vstack([TTA_Classification_All(model_, x_test, batch_size=batch_size)
                                          for model_ in classification_model]), axis=0).flatten()
        else:
            preds_classify = TTA_Classification_All(classification_model, x_test, n_iters=4, batch_size=batch_size).flatten()
    else:
        # ensembling without TTA
        if isinstance(classification_model, (list, tuple)):
            # stacking across the batch_size dimension
            preds_seg = np.mean(np.vstack([model_.predict(x_test, batch_size=batch_size)
                                          for model_ in classification_model]), axis=0).flatten()
        else:
            preds_classify = classification_model.predict(x_test, batch_size=batch_size).flatten()

    test_ids = [Path(fpath).stem for fpath in test_fpaths] # for the df
    # thresholding
    preds_classify[preds_classify >= threshold] = 1
    preds_classify[preds_classify < threshold] = -1
    # converting arr to a list of integers
    preds_classify = [int(pred) for pred in preds_classify.tolist()]
    # creating first df
    sub_df = pd.DataFrame({"ImageId": test_ids, "EncodedPixels": preds_classify})
    sub_df.to_csv("submission_classification.csv", index=False)
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
