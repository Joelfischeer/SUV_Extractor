"""
Organ lists and combination logic per segmentation backend.
Used by main.py and compare_segmentations.py, so both use the same definitions.

Names follow the TotalSegmentator convention; MOOSE names are mapped to it in
Image_loading/segmentation_sources.py (vertebra_L2 -> vertebrae_L2).
"""

ORGANS_OF_INTEREST = [
    'colon',
    'duodenum',
    'heart',
    'small_bowel',
    'kidneys',
    'liver',
    'pancreas',
    'spleen',
    'stomach',
    'thyroid_gland',
    'brain',
    'muscle',
    'aorta',
]

# superior_vena_cava was removed: MOOSE has no such region and it was missing
# for most patients. Organs can be added/removed here freely; an organ that a
# segmentation does not provide gives NaN and is listed in the segmentation
# status (organs_not_available).

# Only in MOOSE (clin_ct_body_composition). TotalSegmentator's 'tissue_types'
# task also produces subcutaneous_fat and skeletal_muscle (but torso_fat, not
# visceral_fat), so these are compared automatically if they exist for both.
MOOSE_EXTRA_ORGANS = ['subcutaneous_fat', 'visceral_fat', 'skeletal_muscle']

COMBINATION_LOGIC = {
    'lungs': ['lung_lower_lobe_left', 'lung_upper_lobe_left',
              'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right'],
    'adrenal_glands': ['adrenal_gland_left', 'adrenal_gland_right'],
    'kidneys': ['kidney_left', 'kidney_right'],
    'muscle': ['gluteus_maximus_right', 'gluteus_maximus_left'],
}

# MOOSE has no single 'heart' / 'thyroid_gland' label
COMBINATION_LOGIC_MOOSE_EXTRA = {
    'heart': ['heart_myocardium', 'heart_atrium_left', 'heart_atrium_right',
              'heart_ventricle_left', 'heart_ventricle_right'],
    'thyroid_gland': ['thyroid_left', 'thyroid_right'],
}


def organ_config(backend):
    """(organs_of_interest, combination_logic) for 'totalsegmentator' or 'moose'."""
    organs = list(ORGANS_OF_INTEREST)
    combination = dict(COMBINATION_LOGIC)
    if backend.lower() == "moose":
        organs += MOOSE_EXTRA_ORGANS
        combination.update(COMBINATION_LOGIC_MOOSE_EXTRA)
    return organs, combination