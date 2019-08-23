import numpy as np
import pandas as pd
import os
import skimage

from tqdm import tqdm
from efficientnet_seg.inference.mask_functions import *

def ensemble_segmentation_from_sub(df_sub_list, min_solutions=3):
    """
    Refactored https://www.kaggle.com/giuliasavorgnan/pneumothorax-models-ensemble-average
    into a reusable function.
    Args:
        df_sub_list (list, tuple): of raw submission DataFrames.
        min_solutions (int): the number of each submissions that must agree for a pixel
            to be considered as pneumothorax (+).
    Returns:
        df_avg_sub (pd.DataFrame): the averaged submission DataFrame
    """
    print("WARNING. This tends to do worse than raw ensembling.")
    iid_list = df_sub_list[0]["ImageId"].unique()
    print("{0} unique image IDs.".format(len(iid_list)))
    assert (min_solutions >= 1 and min_solutions <= len(df_sub_list)), \
        "min_solutions has to be a number between 1 and the number of submission files"
    # create empty final dataframe
    df_avg_sub = pd.DataFrame(columns=["ImageId", "EncodedPixels"])
    df_avg_sub_idx = 0 # counter for the index of the final dataframe

    # iterate over image IDs
    for iid in tqdm(iid_list):
        # initialize prediction mask
        avg_mask = np.zeros((1024,1024))
        # iterate over prediction dataframes
        for df_sub in df_sub_list:
            # extract rles for each image ID and submission dataframe
            rles = df_sub.loc[df_sub["ImageId"]==iid, "EncodedPixels"]
            # iterate over rles
            for rle in rles:
                # if rle is not -1, build prediction mask and add to average mask
                if "-1" not in str(rle):
                    avg_mask += rle2mask(rle, 1024, 1024) / float(len(df_sub_list))
        # threshold the average mask
        avg_mask = (avg_mask >= (min_solutions * 255. / float(len(df_sub_list)))).astype("uint8")
        # extract rles from the average mask
        avg_rle_list = []
        if avg_mask.max() > 0:
            # label regions
            labeled_avg_mask, n_labels = skimage.measure.label(avg_mask, return_num=True)
            # iterate over regions, extract rle, and save to a list
            for label in range(1, n_labels+1):
                avg_rle = mask2rle((255 * (labeled_avg_mask == label)).astype("uint8"), 1024, 1024)
                avg_rle_list.append(avg_rle)
        else:
            avg_rle_list.append("-1")
        # iterate over average rles and create a row in the final dataframe
        for avg_rle in avg_rle_list:
            df_avg_sub.loc[df_avg_sub_idx] = [iid, avg_rle]
            df_avg_sub_idx += 1 # increment index

    df_avg_sub.to_csv("average_submission.csv", index=False)
    return df_avg_sub

def ensemble_classification_from_df(df_path_list, threshold=0.5):
    """
    Reads classification_probabilties.csv files and ensembles them.
    Args:
        df_path_list (list, tuple): of paths to .csv files
            They all must have the same `ImageId`'s and the probabilities should be floats.
    Returns:
        ensembled_df (pd.DataFrame): df with the thresholded classification probabilties
    """
    dfs_list = [pd.read_csv(df_path) for df_path in df_path_list]
    ensembled_dict = {"ImageId": [], "EncodedPixels": []}
    for id_ in tqdm(dfs_list[0]["ImageId"]):
        # ensembling the classification probability for each df at each id_ by averaging
        p_avg = np.mean([float(df.loc[df["ImageId"] == id_, "EncodedPixels"])
                         for df in dfs_list])
        # thresholding
        p_thresholded = 1 if p_avg >= threshold else -1
        ensembled_dict["ImageId"].append(id_), ensembled_dict["EncodedPixels"].append(p_thresholded)
    ensembled_df = pd.DataFrame(ensembled_dict)
    ensemble_csv_path = os.path.join(os.getcwd(), "ensembled_classification.csv")
    ensembled_df.to_csv(ensemble_csv_path, index=False)
    print("Ensembled classification predictions saved at {0}".format(ensemble_csv_path))
    return ensembled_df

def create_classification_p_df(pred_act, test_ids):
    """
    Saves predicted classification probabilities as a csv
    Args:
        pred_arr (np.ndarray): 1D numpy array of predicted probabilities for each image
            corresponding to each test_id.
        test_ids (list): dicom ids corresponding to each predicted probability
    Returns:
        None
    """
    pred_act = pred_act.tolist()
    # creating probability df
    sub_df = pd.DataFrame({"ImageId": test_ids, "EncodedPixels": pred_act})
    sub_df.to_csv("classification_probabilities.csv", index=False)
    print("Classification probabilities saved at {0}".format(os.path.join(os.getcwd(), "classification_probabilities.csv")))
