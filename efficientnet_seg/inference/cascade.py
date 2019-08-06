import glob
import cv2
import skimage
import os
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from tqdm import tqdm

from .classification import *
from .segmentation import *
from .utils import *

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
