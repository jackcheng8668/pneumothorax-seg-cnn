import glob
import pandas as pd

from .classification import *
from .segmentation import *
from .utils import *

def create_submission(classification_model, seg_model, test_fpaths=None, classification_channels=3,
                      seg_channels=3, classification_img_size=256, seg_img_size=256, batch_size=32, tta=True,
                      classification_thresh=0.5, seg_thresh=0.5, classify_csv_fpath=None):
    """
    Performs the cascade. All non-pneumothorax predictions are "-1". All pneumothorax patients
    are then passed to the segmentation model to generate the predicted mask, which is then
    converted to a run-length encoding for the submission file.

    Assuming binary for pneumothorax classification/segmentation.
    """
    if test_fpaths is None:
        test_fpaths = glob.glob('./test/*') # assumes this directory for now
    if classify_csv_fpath is None:
        # Stage 1: Classification predictions
        sub_df = Stage1(classification_model, test_fpaths, channels=classification_channels,
                        img_size=classification_img_size, batch_size=batch_size, tta=tta,
                        threshold=classification_thresh)
    else:
        print("Skipping Stage 1...")
        sub_df = pd.read_csv(classify_csv_fpath)
    # Stage 2: Segmentation
    _ = Stage2(seg_model, sub_df, test_fpaths, channels=seg_channels, img_size=seg_img_size,
               batch_size=batch_size, tta=tta, threshold=seg_thresh)
