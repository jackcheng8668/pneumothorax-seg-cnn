import numpy as np
import pandas as pd

class Oversampler(object):
    """
    For oversampling patients with pneumothorax to balance out the class distribution.
    NOTE: This does a per-patient balancing, not per-pixel. So the # of pneumothorax paitents
    will be balanced with # of non-pneumothorax patients, not the # of pixels that represent
    pneumothorax v. # of pixels that are not part of the pneumothorax.

    Also, this assumes that there are more negative cases than positive cases already.

    Attributes:
        fpaths (list): list of file paths to the dicom files. This will be augmented and passed back to the user.
        rle_csv_path (str): path to `train-rle.csv`
        min_ratio (float): desired minimum ratio of pneumothorax to non-pneumothorax cases. Oversamples until the
        fpaths ratio exceeds or is equal to this ratio.

    Main Method:
        adjust_to_ratio: returns the oversampled and balanced list of fpaths
    """
    def __init__(self, fpaths, rle_csv_path, min_ratio=1., shuffle=True):
        self.fpaths = fpaths
        self.df = pd.read_csv(rle_csv_path, header=None, index_col=0)
        self.min_ratio = min_ratio
        self.shuffle = shuffle

    def adjust_to_ratio(self):
        """
        Oversamples positive class cases (patients with pneumothorax) to balance the distribution.
        Keeps oversampling until the ratio exceeds or is equal to the provided lower bound, min_ratio.

        If the min_ratio is already met, nothing will happen.
        """
        pos, neg = self.group_files_pos_neg_class()
        current_ratio = pos/neg
        print("Current Ratio: {0}".format(current_ratio))
        while current_ratio < self.min_ratio:
            random_pos_fpath = np.random.choice(self.pos_fpaths)
            self.fpaths.append(random_pos_fpath)
            pos+=1
            current_ratio = pos/neg
        print("Class Ratio (class1/class0) After Balancing: {0}".format(current_ratio))
        if self.shuffle:
            np.random.shuffle(self.fpaths)
        return self.fpaths

    def group_files_pos_neg_class(self):
        """
        Method that iterates through the train-rle.csv and groups the patients
        based on their labels into pneumothorax/non-pneumothorax patients.

        Attributes:
            self.pos_fpaths (list):
            self.neg_fpaths (list):
        Returns:
            tuple of the lengths of each list of fpaths (pos/neg)
        """
        self.pos_fpaths = []
        self.neg_fpaths = []
        for fpath in self.fpaths:
            encoded_pixels = self.df.loc[fpath.split('/')[-1][:-4], 1]
            if (type(encoded_pixels) != str or (type(encoded_pixels) == str and encoded_pixels != ' -1')):
                self.pos_fpaths.append(fpath)
            elif encoded_pixels == " -1":
                self.neg_fpaths.append(fpath)

        assert len(self.pos_fpaths)+len(self.neg_fpaths) == len(self.fpaths), "# of filepaths doesn't match with the sum of the # of grouped class filepaths."
        return (len(self.pos_fpaths), len(self.neg_fpaths))
