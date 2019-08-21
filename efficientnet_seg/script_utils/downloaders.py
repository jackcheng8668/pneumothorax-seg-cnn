import os
import json

from google_drive_downloader import GoogleDriveDownloader as gdd
# (fname tuples, file_id)
NIH_WEIGHTS = {"densenet": ("DenseNet169_NIH15_Px448.h5", "1DWCsF3fPkTqHDdgsZSKlcxs90ozFh4hS"),
               "inception": ("InceptionResNetV2_NIH15_Px256.h5", "1ELBeWmK99Jpdd-DXC51qymElj1R_uwMd"),
               "xception":  ("Xception_NIH15_Px320.h5", "1tla9l9y0ap7SSheEdx7-hWmqaU0h9QG3"),
              }
# For 90 10 00 splits
FOLDS = {"1": "1omgOgJk-Mw15ney7q-BpxCX7PBiiWt8n",
         "2": "1CapIIXTv7GnuHa3l0JQT3kEbtHuhViAP",
         "3": "1DOXyEFpvtS1q-A1C9SKq1fc__YgKSqtc",
        }

def download_nih_weights(model_name="densenet"):
    """
    Wrapper to download reuploaded (to a local gdrive) pretrained NIH CNN weights from https://github.com/i-pan/kaggle-rsna18.
    Args:
        model_name (str): one of `densenet`, `inception`, or `xception`
    Returns:
        None
    """
    fname, f_id = NIH_WEIGHTS[model_name]
    print("Weights are from https://github.com/i-pan/kaggle-rsna18")
    fpath = os.path.join(os.getcwd(), fname)
    gdd.download_file_from_google_drive(file_id=f_id, dest_path=fpath,
                                        overwrite=False, unzip=False)

def download_and_open_fold(fold=1):
    """
    Args:
        fold (int): Fold to download
    Returns:
        fpaths_dict (dict): dictionary of `train` and `val` filepaths
    """
    fold = str(int(fold))
    all_folds = list(FOLDS.keys())
    if not fold in all_folds: raise Exception("`fold` must be one of {0}".format(all_folds))
    # downloading the fold .json
    fold_fname = "fold{0}_901000.json".format(fold)
    f_id = FOLDS[fold]
    fpath = os.path.join(os.getcwd(), fold_fname)
    # no need to be redundant and download it again if the .json exists
    if not os.path.exists(fpath):
        gdd.download_file_from_google_drive(file_id=f_id, dest_path=fpath,
                                            overwrite=False, unzip=False)
    print("Loading from json...")
    with open(fold_fname, "r") as fp:
        fpaths_dict = json.load(fp)
    return fpaths_dict
