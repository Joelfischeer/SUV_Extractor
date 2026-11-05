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
    'gluteal_muscle',      # gluteus maximus left + right (COMBINATION_LOGIC), same in all tools
    'aorta',
    'adrenal_glands',      # left + right (COMBINATION_LOGIC); same names in all three tools
    # body composition (skeletal_muscle = all skeletal muscle, separate from
    # gluteal_muscle above)
    #   TotalSegmentator: 'tissue_types' task (licence) - subcutaneous_fat,
    #                     skeletal_muscle, torso_fat (used as visceral_fat)
    #   MOOSE:            clin_ct_body_composition - subcutaneous_fat,
    #                     skeletal_muscle, visceral_fat
    #   CADS:             task 559 - subcutaneous_tissue, muscle (no visceral fat)
    'subcutaneous_fat',
    'skeletal_muscle',
    'visceral_fat',
]

# superior_vena_cava was removed: MOOSE has no such region and it was missing
# for most patients. Organs can be added/removed here freely; an organ that a
# segmentation does not provide gives NaN and is listed in the segmentation
# status (organs_not_available).

# Only in MOOSE (none at the moment)
MOOSE_EXTRA_ORGANS = []

# TotalSegmentator calls the fat inside the trunk 'torso_fat'; it is used as
# 'visceral_fat' (a "combination" of one region works as a renaming). Note:
# torso_fat is all fat inside the trunk (abdomen and thorax); MOOSE's
# visceral_fat may cover only a slab at L3 (clin_ct_body_composition workflow).
COMBINATION_LOGIC_TS_EXTRA = {
    'visceral_fat': ['torso_fat'],
}

COMBINATION_LOGIC = {
    'lungs': ['lung_lower_lobe_left', 'lung_upper_lobe_left',
              'lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right'],
    'adrenal_glands': ['adrenal_gland_left', 'adrenal_gland_right'],
    'kidneys': ['kidney_left', 'kidney_right'],
    # gluteal muscle: gluteus maximus (all three tools also have gluteus_medius_*
    # and gluteus_minimus_* - add them here for all gluteal muscles)
    'gluteal_muscle': ['gluteus_maximus_right', 'gluteus_maximus_left'],
}

# MOOSE has no single 'heart' / 'thyroid_gland' label
COMBINATION_LOGIC_MOOSE_EXTRA = {
    'heart': ['heart_myocardium', 'heart_atrium_left', 'heart_atrium_right',
              'heart_ventricle_left', 'heart_ventricle_right'],
    'thyroid_gland': ['thyroid_left', 'thyroid_right'],
}


# Only in CADS (e.g. 'csf' with task 557 in CADS_TASKS). CADS has single labels
# for heart (task 556) and thyroid (task 558), so no combination is needed; it
# has no visceral fat.
CADS_EXTRA_ORGANS = []


def organ_config(backend):
    """(organs_of_interest, combination_logic) for 'totalsegmentator', 'moose' or 'cads'."""
    organs = list(ORGANS_OF_INTEREST)
    combination = dict(COMBINATION_LOGIC)
    if backend.lower() == "totalsegmentator":
        combination.update(COMBINATION_LOGIC_TS_EXTRA)
    elif backend.lower() == "moose":
        organs += MOOSE_EXTRA_ORGANS
        combination.update(COMBINATION_LOGIC_MOOSE_EXTRA)
    elif backend.lower() == "cads":
        organs += CADS_EXTRA_ORGANS
    return organs, combination