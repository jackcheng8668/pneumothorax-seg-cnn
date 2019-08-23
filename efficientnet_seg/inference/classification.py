import os
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from functools import partial

from efficientnet_seg.io.data_aug import data_augmentation
from efficientnet_seg.inference.utils import load_input, batch_test_fpaths
from efficientnet_seg.io.utils import preprocess_input

def Stage1(classification_model, test_fpaths, channels=3, img_size=256, batch_size=32,
           fpaths_batch_size=320, tta=True, n_tta_iter_per_image=4, tta_then_preprocess=True,
           threshold=0.5, model_name=None, save_p=True, preprocess_fn=None, **kwargs):
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
        fpaths_batch_size (int): number of images to load into memory at a time.
            Adjust this parameter when you're ensembling or doing TTA (memory-intensive).
        tta (boolean): whether or not to apply test-time augmentation.
        n_tta_iter_per_image (int): number of iterations of test-time augmentation with
            `data_aug.data_augmentation`. Defaults to 4.
        tta_then_preprocess (bool): whether or not to preprocess after TTA or preprocess and then TTA.
        threshold (float): threshold for the predicted arrays where
            anything >= threshold = 255 and anything < threshold = 0. Defaults to 0.5.
        model_name (str): Either 'densenet', 'inception', or 'xception' to specify the preprocessing method
        save_p (bool): whether or not to save the classification probabilties. If True, probabilities are saved
            as a .csv file at cwd/classification_probabilties.csv
        preprocess_fn (function): function to preprocess the test arrays with. Specify the other arguments
            with **kwargs. However, it must have the argument for x_test and the argument,`model_name`.
    Returns:
        sub_df (pd.DataFrame): the classification submission data frame (Encoded pixels are 1/-1 for pneumothorax/no pneumothorax).
    """
    # not fatal error, so we just show a quick warning message
    # Can still run alright but it's better to just set tta_then_preprocess=False
    if not tta and tta_then_preprocess:
        print("Warning! Your data will not be preprocessed with `preprocess_fn` as intended \
              because preprocessing will be done AFTER TTA, but TTA is not enabled.")
    # Stage 1: Classification predictions
    print("Commencing Stage 1: Prediction of Pneumothorax or No Pneumothorax Patients")
    # Load test set
    ## batching test_fpaths; # preserves order
    test_fpaths_batched = batch_test_fpaths(test_fpaths, batch_size=fpaths_batch_size)
    print("{0} file batches with sizes {1}".format(len(test_fpaths_batched), [len(batch) for batch in test_fpaths_batched]))
    preds_classify = np.array([])
    for fpaths_batch in test_fpaths_batched:
        x_test = np.asarray([load_input(fpath, img_size, channels=channels) for fpath in fpaths_batch])
        if not tta_then_preprocess:
            # preprocess -> TTA
            # default just converts the input from int -> float vvvvvvvvvvvvvvvvvvv
            preprocess_fn = partial(preprocess_input, model_name=model_name) if preprocess_fn is None else preprocess_fn
            x_test = preprocess_fn(x_test, model_name=model_name, **kwargs)
            # reset to None so it doesn't preprocess again
            ## Not entirely true ^, it does cast the inputs from int -> float32
            ## if need be.
            preprocess_fn = None
        else:
            preprocess_fn = partial(preprocess_input, model_name=model_name) if preprocess_fn is None \
                            else partial(preprocess_fn, model_name=model_name, **kwargs)
        # predictions (with/without TTA)
        preds_classify_batch = run_classification_prediction(x_test, classification_model,
                                                             batch_size=batch_size, tta=tta,
                                                             n_tta_iter_per_image=n_tta_iter_per_image,
                                                             preprocess_fn=preprocess_fn)
        # appending -> flattening
        preds_classify = np.append(preds_classify, preds_classify_batch)
    # creating our df
    test_ids = [Path(fpath).stem for fpath in test_fpaths] # for the df
    if save_p:
        from efficientnet_seg.inference.ensemble_df import create_classification_p_df
        _ = create_classification_p_df(preds_classify, test_ids)
    sub_df = create_thresholded_classification_csv(preds_classify, test_ids, threshold=threshold)
    print("Stage 1 Completed.")
    return sub_df

def create_thresholded_classification_csv(pred_arr, test_ids, threshold=0.5):
    """
    Creates the thresholded version of the classification predictions data frame (csv).
    Args:
        pred_arr (np.ndarray): 1D numpy array of predicted probabilities for each image
            corresponding to each test_id.
        test_ids (list): dicom ids corresponding to each predicted probability
        threshold (float): threshold
    Returns:
        None
    """
    # thresholding
    pred_arr[pred_arr >= threshold] = 1
    pred_arr[pred_arr < threshold] = -1
    # converting arr to a list of integers
    preds_classify = [int(pred) for pred in pred_arr.tolist()]
    # creating first df
    sub_df = pd.DataFrame({"ImageId": test_ids, "EncodedPixels": preds_classify})
    save_path = os.path.join(os.getcwd(), "submission_classification.csv")
    sub_df.to_csv(save_path, index=False)
    print("Classification csv saved at {0}.".format(save_path))
    return sub_df

def run_classification_prediction(x_test, classification_model, batch_size=32, tta=True,
                                  n_tta_iter_per_image=4, preprocess_fn=None):
    """
    Handles raw classification model prediction. Supports TTA and ensembling.
    Args:
        x_test (np.ndarray): shape (n, x, y, n_channels)
        classification_model (a single tf.keras.model.Model or keras.model.Model or a list of them): assumes
            that they all need the same input. When `classification_model` is a list/tuple, the models are
            ensembled (predictions are averaged.)
        batch_size (int): model prediction batch size
        tta (boolean): whether or not to apply test-time augmentation.
        n_tta_iter_per_image (int): number of iterations of test-time augmentation with
            `data_aug.data_augmentation`. Defaults to 4.
        preprocess_fn (function): function to preprocess the test arrays with. It should only have
            one argument, the input array. To use more arguments, use functools.partial.
    Returns:
        preds_classify (np.ndarray): shape (n, 1); assumes prediction channel is 1.
    """
    ## Hacky fix for binary cases where the output is (N, 1)
    ### Prevents lists being saved as nested lists
    if tta:
        # ensembling with TTA
        if isinstance(classification_model, (list, tuple)):
            # stacking across the batch_size dimension
            preds_classify = [TTA_Classification_All(model_, x_test, n_iter=n_tta_iter_per_image,
                              batch_size=batch_size, preprocess_fn=preprocess_fn)
                              for model_ in classification_model]
            # ensembling by averaging
            preds_classify = np.mean(np.stack(preds_classify), axis=0).flatten()
        else:
            preds_classify = TTA_Classification_All(classification_model, x_test, n_iter=n_tta_iter_per_image,
                                                    batch_size=batch_size, preprocess_fn=preprocess_fn).flatten()
    else:
        # ensembling without TTA
        if isinstance(classification_model, (list, tuple)):
            # stacking across the batch_size dimension
            preds_classify = np.mean(np.stack([model_.predict(x_test, batch_size=batch_size)
                                               for model_ in classification_model]), axis=0).flatten()
        else:
            preds_classify = classification_model.predict(x_test, batch_size=batch_size).flatten()
    return preds_classify

def TTA_Classification_All(model, test_arrays, n_iter=4, batch_size=32, seed=88, preprocess_fn=None):
    """
    Thin wrapper around for TTA Classification for TTA on all images instead of per image.
    """
    print("TTA for classification...")
    return np.asarray([TTA_Classification(model, test_arr, n_iter=n_iter, batch_size=batch_size,
                       seed=seed, preprocess_fn=preprocess_fn) for test_arr in tqdm(test_arrays)])

def TTA_Classification(model, test_array, n_iter=4, batch_size=32, seed=88, preprocess_fn=None):
    """
    Test-time augmentation with array inversion, horizontal and vertical flipping,
    gaussian smoothing, random rotations, and random zooms.
    NOTE: THIS IS PER-IMAGE BECAUSE OF THE MEMORY LIMITATIONS.
    Args:
        model (instance of keras.models.Model): should predict an array with shape (N, n_classes)
        test_array (np.ndarray): shape of (H, W, C)
        batch_size: the batch size for prediction
        seed (int): desired seed for the data aug
        preprocess_fn (function): function to preprocess the test arrays with. It should only have
            one argument, the input array. To use more arguments, use functools.partial.
    Returns:
        preds_test (np.ndarray): predicted probability for test_array (n_classes,)
    """
    # for reproducibility
    np.random.seed(seed)
    # creating arrays to store the augmented images
    original_img_array = np.empty((n_iter+1, test_array.shape[0], test_array.shape[1], test_array.shape[2]))
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
    for each_iter in range(n_iter):
        original_img_array[each_iter+1] = data_augmentation(original_img)[0] # no mask
        inverted_img_array[each_iter+1] = data_augmentation(inverted_img)[0] # no mask
        hflipped_img_array[each_iter+1] = data_augmentation(hflipped_img)[0] # no mask
    tmp_array = np.vstack((original_img_array, inverted_img_array, hflipped_img_array))
    # preprocessing
    if preprocess_fn is not None:
        tmp_array = preprocess_fn(tmp_array)

    n_classes = int(model.get_output_at(-1).get_shape()[1])
    prediction = np.mean(model.predict(tmp_array), axis=0)
    return prediction
