
def compute_suv(
    organ_dict,
    organs_of_interest,
    patient_id=None
):
    """
    Compute mean SUV per organ from PET image. 
    Ignore values outside the segmentations.
    
    """
    
    import numpy as np

    result = {"Patient": patient_id}
    
    for organ in organs_of_interest:

        if organ not in organ_dict:
            result[organ] = np.nan
            continue
        
        organ_non_zero = organ_dict[organ][organ_dict[organ] > 0]

        if len(organ_non_zero) == 0:
            result[organ] = np.nan
            continue
        
        mean_suv = np.mean(organ_non_zero)
        result[f'{organ}_mean_automatic'] = round(mean_suv, 3)

        max_suv = max(organ_non_zero)
        result[f'{organ}_max_automatic'] = round(max_suv, 3)

        min_suv = min(organ_non_zero)
        result[f'{organ}_min_automatic'] = round(min_suv, 3)
    
    return result


