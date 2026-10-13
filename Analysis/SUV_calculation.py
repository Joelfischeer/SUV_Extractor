def compute_suv(
    organ_dict,
    organs_of_interest,
    patient_id=None
):
    """
    Compute mean, max and min SUV per organ from the cropped PET volumes.
    Ignore values outside the segmentations (voxels == 0).

    Missing or empty organs get NaN in the same columns
    ({organ}_mean/max/min_automatic), so every patient has the same columns.
    """

    import numpy as np

    result = {"Patient": patient_id}

    for organ in organs_of_interest:
        keys = (f'{organ}_mean_automatic', f'{organ}_max_automatic', f'{organ}_min_automatic')

        organ_non_zero = (organ_dict[organ][organ_dict[organ] > 0]
                          if organ in organ_dict else np.array([]))

        if organ_non_zero.size == 0:
            for k in keys:
                result[k] = np.nan
            continue

        result[keys[0]] = round(float(np.mean(organ_non_zero)), 3)
        result[keys[1]] = round(float(np.max(organ_non_zero)), 3)
        result[keys[2]] = round(float(np.min(organ_non_zero)), 3)

    return result