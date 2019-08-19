import pydicom
import pandas as pd
from tqdm import tqdm

def extract_metadata_into_csv(fpaths_list, train_rle_df, dset_dir, save_path="metadata.csv"):
    """
    Extracts metadata in the .dcm files and organizes it into a .csv / DataFrame. Source
    code for useful_files/test_metadata.csv and useful_files/train_metadata.csv.

    Refactored: https://www.kaggle.com/retyidoro/eda-of-pneumothorax-dataset. Also,
    make sure to use the pneumothorax.zip (from the outputs of the above kernel) for
    this extraction.

    Args:
        fpaths_list (list): of file paths to the .dcm files
        train_rle_df (pd.DataFrame):
        dset_dir (str): Path to the pneumothorax dataset
        save_path (str): where to save the corresponding metadata dataframe
    Returns:
        a pd.DataFrame containing metadata
    """
    missing = 0
    patients = []
    for fpath in tqdm(fpaths_list):
        data = pydicom.read_file(str(fpath))
        patient = {}
        # appending metadata dictionaries to list
        patient["UID"] = data.SOPInstanceUID
        try:
            encoded_pixels = train_rle_df[train_rle_df["ImageId"] == patient["UID"]].values[0][1]
            patient["EncodedPixels"] = encoded_pixels
        except:
            missing = missing + 1
        patient["Age"] = data.PatientAge
        patient["Sex"] = data.PatientSex
        patient["Modality"] = data.Modality
        patient["BodyPart"] = data.BodyPartExamined
        patient["ViewPosition"] = data.ViewPosition
        dicom_fpath = "{0}\\dicom-images-train\\{1}\\{2}\\{3}.dcm".format(dset_dir, data.StudyInstanceUID, \
                                                                          data.SeriesInstanceUID, data.SOPInstanceUID)
        patient["path"] = dicom_fpath
        patients.append(patient)

    print("missing labels: {0}".format(missing))
    #pd.set_option('display.max_colwidth', -1)
    df_patients = pd.DataFrame(patients, columns=["UID", "EncodedPixels", "Age", "Sex", "Modality", "BodyPart", "ViewPosition", "path"])
    print("images with labels: {0}".format(df_patients.shape[0]))
    df_patients.to_csv(save_path)
    return df_patients
