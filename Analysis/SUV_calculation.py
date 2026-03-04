import os
import numpy as np
import pandas as pd

def compute_suv(
    all_patients_crops,
    organs_of_interest
):
    """
    Parameters
    ----------
    all_patients_crops : dict
        {
            patient_id: {
                "liver": np.array,
                "spleen": np.array,
                "aorta": np.array
            },
            ...
        }

    organs_of_interest : list
        Organs to compute SUV for (excluding aorta)

    output_csv_path : str
        Where to save CSV
    """

    results = []

    for patient, organ_dict in all_patients_crops.items():

        if "aorta" not in organ_dict:
            print(f"⚠️ No aorta for {patient}, skipping.")
            continue

        # --------------------------------------------------
        # Compute mean SUV of aorta (non-zero only)
        # --------------------------------------------------
        aorta_values = organ_dict["aorta"]
        aorta_non_zero = aorta_values[aorta_values > 0]

        if len(aorta_non_zero) == 0:
            print(f"⚠️ Aorta empty for {patient}, skipping.")
            continue

        mean_aorta_suv = np.mean(aorta_non_zero)

        patient_result = {"Patient": patient}

        # --------------------------------------------------
        # Compute normalized SUV for each organ
        # --------------------------------------------------
        for organ in organs_of_interest:

            if organ not in organ_dict:
                patient_result[organ] = np.nan
                continue

            organ_values = organ_dict[organ]
            organ_non_zero = organ_values[organ_values > 0]

            if len(organ_non_zero) == 0:
                patient_result[organ] = np.nan
                continue

            mean_organ_suv = np.mean(organ_non_zero)

            # Normalize by aorta SUV
            normalized_suv = mean_organ_suv / mean_aorta_suv

            patient_result[organ] = normalized_suv

        results.append(patient_result)
    
    return results


