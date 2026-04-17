def aorta_normalization(PET_organs_raw):

    """
    Compute mean SUV in aorta restricted to L1 vertebrae z-range,
    only using voxels where aorta > 0 (excluding background).
    Falls back to full aorta if needed.
    Assumes PET_organs_raw contains cropped organ 'PET' images with the SUV array.
    """

    import numpy as np

    normalize_value = None

    has_aorta = "aorta" in PET_organs_raw
    has_L1 = "vertebrae_L1" in PET_organs_raw 

    cropped_aorta = PET_organs_raw["aorta"]

    if has_aorta and has_L1:
        aorta_mask = PET_organs_raw["aorta"].copy()
        l1_mask = PET_organs_raw["vertebrae_L1"].copy()

        non_zero_L1 = np.argwhere(l1_mask > 0)
        if non_zero_L1.size > 0:
            min_z_l1, _, _ = non_zero_L1.min(axis=0)
            max_z_l1, _, _ = non_zero_L1.max(axis=0)

            aorta_L1_mask = aorta_mask.copy()
            aorta_L1_mask[:min_z_l1] = 0
            aorta_L1_mask[max_z_l1 + 1:] = 0

            valid_voxels = aorta_L1_mask > 0
            if np.any(valid_voxels):
                normalize_value = float(aorta_L1_mask[valid_voxels].mean())
            else:
                print("⚠️ No voxels >0 in aorta-L1 region, falling back to full aorta.")

        else:
            print("⚠️ No L1 mask, using full cropped aorta.")

    if (normalize_value is None or normalize_value <= 0) and has_aorta:
        valid_voxels_aorta_full = cropped_aorta > 0
        if np.any(valid_voxels_aorta_full > 0):
            normalize_value = float(cropped_aorta[valid_voxels_aorta_full > 0].mean())
        else:
            print("⚠️ No voxels >0 in cropped aorta, no normalization possible.")


    return normalize_value