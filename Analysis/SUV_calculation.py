
def compute_suv(
    organ_dict,
    organs_of_interest,
    patient_id=None
):
    """Compute mean SUV per organ from PET image."""
    
    import numpy as np

    
    result = {"Patient": patient_id}
    
    # Loop through organs:
    for organ in organs_of_interest:

        #If the segmentation is empty, return NaN:
        if organ not in organ_dict:
            result[organ] = np.nan
            continue
        
        #Ignore values outside the organ segmentation which are 0:
        organ_non_zero = organ_dict[organ][organ_dict[organ] > 0]

        #If all values are 0 return NaN:
        if len(organ_non_zero) == 0:
            result[organ] = np.nan
            continue
        
        #Calculate the mean and round to 3 digits:
        mean_suv = np.mean(organ_non_zero)
        result[organ] = round(mean_suv, 3)
    
    return result


