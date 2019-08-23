import numpy as np
import pandas as pd
import cv2
import os
from tqdm import tqdm
from pathlib import Path
from functools import partial

from efficientnet_seg.inference.mask_functions import mask2rle
from efficientnet_seg.inference.utils import load_input, batch_test_fpaths
from efficientnet_seg.io.utils import preprocess_input
from efficientnet_seg.inference.segmentation import TTA_Segmentation_All, zero_out_thresholded_single

def SegmentationOnlyInference(seg_model, test_fpaths, channels=3, img_size=256, batch_size=32,
                              fpaths_batch_size=320, tta=True, threshold=0.5, zero_out_small_pred=True,
                              preprocess_fn=None, **kwargs):
    """
    For segmentation-only pipelines.

    Args:
        seg_model (a single tf.keras.model.Model or keras.model.Model or a list of them): assumes
            that they all need the same input. When `seg_model` is a list/tuple, the models are
            ensembled (predictions are averaged.)
        sub_df (pd.DataFrame): A classification submission dataframe.
        test_fpaths (list or tuple): of file paths to the test images
        channels (int): The number of input channels. Defaults to 3.
        img_size (int): The size of each square input image. Defaults to 256.
        batch_size (int): model prediction batch size
        fpaths_batch_size (int): number of images to load into memory at a time.
            Adjust this parameter when you're ensembling or doing TTA (memory-intensive).
        tta (boolean): whether or not to apply test-time augmentation.
        threshold (float): Value to threshold the predicted probabilities at
        zero_out_small_pred (bool): whether or not to zero out the smaller predicted ROIs.
        preprocess_fn (function): function to preprocess the test arrays with. Specify the other arguments
            with **kwargs.
    Returns:
        sub_df (pd.DataFrame): submission dataframe
    """
    # default just converts the input from int -> flaot
    preprocess_fn = partial(preprocess_input, model_name=None) if preprocess_fn is None else preprocess_fn
    # Stage 2: Segmentation
    print("Commencing the Segmentation of All Test Patients...")
    # assuming full dataset cannot fit into memory
    ## batching test_fpaths; # preserves order
    test_fpaths_batched = batch_test_fpaths(test_fpaths, batch_size=fpaths_batch_size)
    rles = []
    for fpaths_batch in test_fpaths_batched:
        x_test = np.asarray([load_input(fpath, img_size, channels=channels)
                             for fpath in fpaths_batch])
        x_test = preprocess_fn(x_test, **kwargs)

        preds_seg = run_seg_prediction(x_test, seg_model, batch_size=batch_size, tta=tta)
        # resizing -> threhold -> zero out small roi -> transpose + set 1s to 255 + type convert
        print("Converting predictions to the submission data frame...")
        for pred in tqdm(preds_seg):
            # resizing probability maps
            h_w = (pred.shape[0], pred.shape[1])
            if h_w != (1024, 1024):
                # resizing predictions if necessary
                arr = cv2.resize(pred, (1024, 1024))
            # thresholding to do zeroing out
            arr[arr >= threshold] = 1
            arr[arr < threshold] = 0
            if zero_out_small_pred:
                arr = zero_out_thresholded_single(arr)
            # converting to rgb (int, 0-255) and transposing
            arr = (arr.T*255).astype(np.uint8)
            rles.append(mask2rle(arr, 1024, 1024))
    # creating list of str ids (fname without the .dicom or .png)
    test_ids = [Path(fpath).stem for fpath in test_fpaths]
    sub_df = create_sub_from_rles(rles, test_ids)
    sub_df.to_csv("submission_final.csv", index=False)
    print("Done!")
    return sub_df

def create_sub_from_rles(rles, test_ids):
    """
    Creates the submission file from rles.
    Args:
        rles (list): of run-length encodings from mask2rle
        test_ids (list): dicom ids corresponding to each predicted mask
    Returns:
        None
    """
    # creating segmentation rle df
    sub_df = pd.DataFrame({"ImageId": test_ids, "EncodedPixels": rles})
    # handling empty masks
    save_path = os.path.join(os.getcwd(), "submission_segmentation_only.csv")
    sub_df.to_csv(save_path, index=False)
    print("Segmentation-only csv saved at {0}".format(save_path))
    return sub_df

def run_seg_prediction(x_test, seg_model, batch_size=32, tta=True):
    """
    Handles raw model prediction. Supports TTA and ensembling.
    Args:
        x_test (np.ndarray): shape (n, x, y, n_channels)
        seg_model (a single tf.keras.model.Model or keras.model.Model or a list of them): assumes
            that they all need the same input. When `seg_model` is a list/tuple, the models are
            ensembled (predictions are averaged.)
        batch_size (int): model prediction batch size
        tta (boolean): whether or not to apply test-time augmentation.
    Returns:
        preds_seg (np.ndarray): shape (n, x, y); assumes prediction channel is 1, which is squeezed.
    """
    # squeezes are for removing the output classes dimension (1, because binary and sigmoid)
    if tta:
        # ensembling with TTA
        if isinstance(seg_model, (list, tuple)):
            # stacking across the batch_size dimension
            print("Ensembling the models with TTA...")
            preds_seg = np.mean(np.stack([TTA_Segmentation_All(model_, x_test, batch_size=batch_size)
                                          for model_ in tqdm(seg_model)]), axis=0).squeeze()
        else:
            preds_seg = TTA_Segmentation_All(seg_model, x_test, batch_size=batch_size).squeeze()
    else:
        # ensembling without TTA
        if isinstance(seg_model, (list, tuple)):
            # stacking across the batch_size dimension
            print("Ensembling the models...")
            preds_seg = np.mean(np.stack([model_.predict(x_test, batch_size=batch_size)
                                          for model_ in tqdm(seg_model)]), axis=0).squeeze()
        else:
            preds_seg = seg_model.predict(x_test, batch_size=batch_size).squeeze()
    return preds_seg
