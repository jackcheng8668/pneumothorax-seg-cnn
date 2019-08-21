import shutil
import os
import json
import pandas as pd
import numpy as np

from glob import glob
from pathlib import Path
from sklearn.model_selection import train_test_split
from os.path import join
from PIL import Image
from tqdm import tqdm


def create_fold_and_move(train_dir, save_dir, mask_df, fold=1, split_seed=10, adjust_n_files=True):
    """
    Workflow:
        Creates train/val split -> Moves files to separate directories ->
        Creates fold json using those moved directories
    Args:
        train_dir (str): path to the repacked data's training directory (pre-moving files)
            i.e. "/content/train"
        save_dir (str): path to the directory to move all of the training/validation directories to
        mask_df (pd.DataFrame): from `create_mask_df`
        fold (int): Fold to create. Only matters for the output .json filename.
        split_seed (int): Seed for train_test_split.
        adjust_n_files (bool): whether or not to adjust the number of files in the
            fold dictionary to a nice number for clean batch sizes.
    Returns:
        fold_dict (dict): dictionary of `train` and `val` filepaths
    """
    train_fn, val_fn = create_train_val_split(train_dir, mask_df, split_seed=10)
    move_files_post_split(train_fn, val_fn, save_dir)
    # for nicer batch sizes
    train_im_path, val_im_path = join(save_dir, "keras_im_train"), join(save_dir, "keras_im_val")
    # creating train/val fpaths lists for the new train/val directories
    train_fpaths, val_fpaths = glob(join(train_im_path, "*")), glob(join(val_im_path, "*"))

    if adjust_n_files:
        train_fpaths = train_fpaths[:-7] # 9600
        val_fpaths = val_fpaths[:-12] # 1056
    fold_dict = {"train": train_fpaths, "val": val_fpaths}

    print("No. of train files: {0}".format(len(train_fpaths)))
    print("No. of val files: {0}".format(len(val_fpaths)))

    fold_fname = "fold{0}_901000.json".format(fold)
    # saving the fold .json if it doesn't already exist
    save_to_json(fold_dict, fold_fname)

    return fold_dict

def move_files_post_split(train_fn, val_fn, save_dir):
    """
    Specific function to move files to separate train/val directories for both
    images and masks after splitting. These directories are: keras_im_train, keras_im_val,
    keras_mask_train, keras_mask_val.
    Args:
        train_fn (list): list of training filepaths before moving. Refer to the `train_fn` output of `create_train_val_split`.
        val_fn (list): list of validation filepaths before moving. Refer to the `val_fn` output of `create_train_val_split`
        save_dir (str): path to the directory to move all of the training/validation directories to
    Returns:
        None
    """
    # Moving files to their appropriate directories
    # assuming all training/validation files have the same base directory
    # path to everything besides the train/*.dcm itself
    base_dir = str(Path(train_fn[0]).parents[1])
    train_dir, masks_dir= os.path.join(base_dir, "train"), os.path.join(base_dir, "masks")

    # creating the mask filepaths using the training file paths (just replace the train with masks)
    masks_train_fn = [fn.replace(train_dir, masks_dir) for fn in train_fn]
    masks_val_fn = [fn.replace(train_dir, masks_dir) for fn in val_fn]
    # actually moving the files
    train_im_path, train_mask_path = join(save_dir, "keras_im_train"), join(save_dir, "keras_mask_train")
    val_im_path, val_mask_path = join(save_dir, "keras_im_val"), join(save_dir, "keras_mask_val")
    dirs = [train_im_path, train_mask_path, val_im_path, val_mask_path]
    fns = [train_fn, masks_train_fn, val_fn, masks_val_fn]
    print("Moving files to the appropriate directories: {0}".format(dirs))
    for directory, fn_list in zip(dirs, fns):
        move_to_dir(directory, fn_list)

def create_train_val_split(train_dir, mask_df, split_seed=10):
    """
    Creates 90/10 train/val fold to prepare for file and fold .json creation.
    Args:
        train_dir (str): path to the repacked data's training directory (pre-moving files)
        mask_df (pd.DataFrame): from `create_mask_df`
        split_seed (int): Seed for train_test_split.
    Returns:
        tuple (train_fn, val_fn) of lists of filepaths after `train_test_split`
    """
    # setting up the file paths
    all_train_fn = glob(os.path.join(train_dir, "*"))

    train_fn, val_fn = train_test_split(all_train_fn, stratify=mask_df.labels, test_size=0.1, random_state=split_seed)

    print("No. of train files: {0}".format(len(train_fn)))
    print("No. of val files: {0}".format(len(val_fn)))

    return (train_fn, val_fn)

def create_mask_df(masks_dir):
    """
    Creates a DataFrame with some EDA about the labels. This df is mainly used for
    the stratify argument in `train_test_split` for `create_train_val_split`.
    Args:
        masks_dir (str): path to the masks (pre-split)
    Returns:
        mask_df: pandas DataFrame with info on the percentage of pneumothorax in the dataset
    """
    all_mask_fn = glob(os.path.join(masks_dir, "*"))
    mask_df = pd.DataFrame()
    mask_df["file_names"] = all_mask_fn
    mask_df["mask_percentage"] = 0
    mask_df.set_index("file_names", inplace=True)
    for fn in tqdm(all_mask_fn):
        mask_df.loc[fn, "mask_percentage"] = np.array(Image.open(fn)).sum()/(256*256*255) #255 is bcz img range is 255
    mask_df.reset_index(inplace=True)
    mask_df["labels"] = 0
    mask_df.loc[mask_df.mask_percentage>0,"labels"] = 1
    return mask_df

def move_to_dir(directory, base_fns):
    """
    Moves files to a new directory
    """
    os.mkdir(directory)
    for full_fn in base_fns:
        fn = Path(full_fn).name
        shutil.move(full_fn, os.path.join(directory, fn))

def save_to_json(file, fpath):
    """
    Saves a file to fpath (the file path to a .json file).
    """
    print("Saving {0}".format(fpath))
    with open(fpath, "w") as fp:
        json.dump(file, fp)

def load_json(fpath):
    """
    Loads a .json file from fpath. Used to load the fold.json files.
    """
    print("Loading {0}".format(fpath))
    with open(fpath, "r") as fp:
        fpaths_dict = json.load(fp)
    return fpaths_dict
