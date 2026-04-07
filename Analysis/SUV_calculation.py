import os
from pathlib import Path
import numpy as np
import pandas as pd
import SimpleITK as sitk
import pydicom
import math


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
            
        organ_non_zero = organ_dict[organ][organ_dict[organ] > 0] #Ignore noise under 0.5
        if len(organ_non_zero) == 0:
            result[organ] = np.nan
            continue
            
        mean_suv = np.mean(organ_non_zero)
        result[organ] = round(mean_suv, 3)  # Already aorta-normalized!
    
    return result


