import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import math


def convert_pet_to_suv(pet_folder: Path):
    """Load PET series and convert BQML to SUVbw."""
    
    # Pick first DICOM for metadata
    dicom_file = list(pet_folder.glob("*.dcm"))[0]
    ds = pydicom.dcmread(dicom_file)
    
    #Read metadata from file:
    weight_g = ds.PatientWeight * 1000  # we need grams
    rad_info = ds.RadiopharmaceuticalInformationSequence[0] # Tracer info
    injected_dose = rad_info.RadionuclideTotalDose  # Bq
    half_life = rad_info.RadionuclideHalfLife       # s

    #Convert the DICOM timeformat into seconds:
    def dicom_time_to_seconds(t):
        return int(t[:2])*3600 + int(t[2:4])*60 + float(t[4:])
    
    #Calculate time difference between injection and scan aquisition:
    delta_t = dicom_time_to_seconds(ds.AcquisitionTime) - dicom_time_to_seconds(rad_info.RadiopharmaceuticalStartTime)

    #Calculate decay constant:
    lambda_decay = math.log(2) / half_life

    #Correct injected dose for decay: (Get remaining activity at scan time)
    decay_corrected_dose = injected_dose * math.exp(-lambda_decay * delta_t)
    
    # Load full PET series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(pet_folder))
    reader.SetFileNames(dicom_names)
    pet_sitk = reader.Execute()
    pet_array = sitk.GetArrayFromImage(pet_sitk).astype(np.float32)
    
    # Convert to SUV
    suv_array = pet_array * weight_g / decay_corrected_dose
    return suv_array


def compute_suv(
    organ_dict,
    organs_of_interest,
    patient_id=None
):
    """Compute mean SUV per organ, normalized by aorta SUV."""
    
    if "aorta" not in organ_dict:
        print(f"⚠️ No aorta for {patient_id}, skipping.")
        return None

    #Get the aorta mean SUV:
    aorta_non_zero = organ_dict["aorta"][organ_dict["aorta"] > 0]
    if len(aorta_non_zero) == 0:
        print(f"⚠️ Aorta empty for {patient_id}, skipping.")
        return None
    mean_aorta_suv = np.mean(aorta_non_zero)

    result = {"Patient": patient_id}
    
    #Loop through organs to normalize by aorta:
    for organ in organs_of_interest:
        if organ not in organ_dict:
            result[organ] = np.nan
            continue
        organ_non_zero = organ_dict[organ][organ_dict[organ] > 0]
        if len(organ_non_zero) == 0:
            result[organ] = np.nan
            continue
        mean_suv = np.mean(organ_non_zero)
        result[organ] = round(mean_suv / mean_aorta_suv, 3)
    
    return result


