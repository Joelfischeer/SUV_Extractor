import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import math


def convert_pet_to_suv(pet_folder: Path):
    """Load PET series and convert to Bq/ml with per-slice RescaleSlope/Intercept."""
    
    # Metadata from first DICOM
    dicom_files = list(pet_folder.glob("*.dcm"))
    ds = pydicom.dcmread(dicom_files[0], force=True)
    weight_g = ds.PatientWeight * 1000
    rad_info = ds.RadiopharmaceuticalInformationSequence[0]
    injected_dose = rad_info.RadionuclideTotalDose
    half_life = rad_info.RadionuclideHalfLife
    
    def dicom_time_to_seconds(t):
        return int(t[:2])*3600 + int(t[2:4])*60 + float(t[4:])
    
    delta_t = dicom_time_to_seconds(ds.AcquisitionTime) - dicom_time_to_seconds(rad_info.RadiopharmaceuticalStartTime)
    lambda_decay = math.log(2) / half_life
    decay_corrected_dose = injected_dose * math.exp(-lambda_decay * delta_t)
    
    # Load full PET series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(pet_folder))
    reader.SetFileNames(dicom_names)
    pet_sitk = reader.Execute()
    
    # ✅ PER-SLICE RescaleSlope + Intercept → Bq/ml
    pet_array = np.zeros(sitk.GetArrayFromImage(pet_sitk).shape, dtype=np.float32)
    for i, fname in enumerate(dicom_names):
        slice_ds = pydicom.dcmread(fname, force=True)
        slope = float(slice_ds.RescaleSlope) if 'RescaleSlope' in slice_ds else 1.0
        intercept = float(slice_ds.RescaleIntercept) if 'RescaleIntercept' in slice_ds else 0.0
        
        slice_raw = sitk.GetArrayFromImage(pet_sitk)[i,:,:]
        slice_calibrated = slice_raw * slope + intercept  # Bq/ml ✓
        pet_array[i,:,:] = slice_calibrated
    
    # Convert Bq/ml → SUV
    suv_array = pet_array * weight_g / decay_corrected_dose
    return suv_array


def compute_suv(
    organ_dict,
    organs_of_interest,
    patient_id=None
):
    """Compute mean SUV per organ from PRE-NORMALIZED PET image."""
    
    result = {"Patient": patient_id}
    
    # Loop through organs - NO normalization needed (already done on full PET!)
    for organ in organs_of_interest:
        if organ not in organ_dict:
            result[organ] = np.nan
            continue
            
        organ_non_zero = organ_dict[organ][organ_dict[organ] > 0]
        if len(organ_non_zero) == 0:
            result[organ] = np.nan
            continue
            
        mean_suv = np.mean(organ_non_zero)
        result[organ] = round(mean_suv, 3)  # Already aorta-normalized!
    
    return result


