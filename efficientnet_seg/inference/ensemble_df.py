import numpy as np
import pandas as pd
import os

from tqdm import tqdm

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
    ensembled_df.to_csv(ensemble_csv_path)
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
