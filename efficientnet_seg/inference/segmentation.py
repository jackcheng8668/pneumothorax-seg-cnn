import numpy as np
import cv2
from tqdm import tqdm
from pathlib import Path
from efficientnet_seg.inference.mask_functions import *
from efficientnet_seg.inference.utils import load_input

def Stage2(seg_model, sub_df, test_fpaths, channels=3, img_size=256, batch_size=32, tta=True,
           threshold=0.5):
    """
    For the second (segmentation) stage of the classification/segmentation cascade. It assumes that the
    seg_model was trained on pos-only examples.

    Args:
        seg_model (a single tf.keras.model.Model or keras.model.Model or a list of them): assumes
            that they all need the same input. When `seg_model` is a list/tuple, the models are
            ensembled (predictions are averaged.)
        sub_df (pd.DataFrame): A classification submission dataframe.
        test_fpaths (list or tuple): of file paths to the test images
        channels (int): The number of input channels. Defaults to 3.
        img_size (int): The size of each square input image. Defaults to 256.
        batch_size (int): model prediction batch size
        tta (boolean): whether or not to apply test-time augmentation.
    """
    # Stage 2: Segmentation
    print("Commencing Stage 2: Segmentation of Predicted Pneumothorax (+) Patients")
    # extracting positive only predictions
    test_ids = np.array([Path(fpath).stem for fpath in test_fpaths])
    seg_ids = test_ids[np.where(sub_df["EncodedPixels"] == 1)[0]].tolist()
    x_test = np.asarray([load_input(fpath, img_size, channels=channels)
                         for fpath in test_fpaths if Path(fpath).stem in seg_ids])
    # squeezes are for removing the output classes dimension (1, because binary and sigmoid)
    if tta:
        # ensembling with TTA
        if isinstance(seg_model, (list, tuple)):
            # stacking across the batch_size dimension
            preds_seg = np.mean(np.vstack([TTA_Segmentation_All(model_, x_test, batch_size=batch_size)
                                          for model_ in seg_model]), axis=0).squeeze()
        else:
            preds_seg = TTA_Segmentation_All(seg_model, x_test, batch_size=batch_size).squeeze()
    else:
        # ensembling without TTA
        if isinstance(seg_model, (list, tuple)):
            # stacking across the batch_size dimension
            preds_seg = np.mean(np.vstack([model_.predict(x_test, batch_size=batch_size)
                                          for model_ in seg_model]), axis=0).squeeze()
        else:
            preds_seg = seg_model.predict(x_test, batch_size=batch_size).squeeze()

    preds_seg[preds_seg >= threshold] = 255
    preds_seg[preds_seg < threshold] = 0
    h_w = (preds_seg.shape[1], preds_seg.shape[2])
    if h_w != (1024, 1024):
        print("Resizing the predictions...")
        preds_seg = np.asarray([cv2.resize(pred, (1024, 1024), interpolation=cv2.INTER_NEAREST).T.astype(np.uint8)
                                for pred in tqdm(preds_seg)])
    sub_df = edit_classification_df(sub_df, preds_seg, seg_ids)
    sub_df.to_csv("submission_final.csv", index=False)

    print("Stage 2 Completed.")

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

def edit_classification_df(df, preds_seg, p_ids):
    """
    Edits in the segmentation rles into the original classification-only submission.
    Args:
        df (pd.DataFrame): with columns, `ImageId` and `EncodedPixels`
        preds_seg (np.ndarray): shape (n, 1024, 1024)
        p_ids (list): list of patient ids that presumably have pneumothorax
    Returns:
        df: The final dataframe to be saved.
    """
    print("Updating the dataframe with the predicted rle's...")
    rles = [mask2rle(pred, 1024, 1024) for pred in tqdm(preds_seg)]
    # injecting the run length encodings
    for id_, rle in zip(p_ids, rles):
        df.loc[df["ImageId"] == id_, "EncodedPixels"] = rle
    # handling empty masks
    df.loc[df.EncodedPixels=="", "EncodedPixels"] = "-1"
    return df
