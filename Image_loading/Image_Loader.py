
def PET_Organ_Cropper(
    data_dir,
    patient,
    organs_of_interest,
    combination_logic=None
):
    """
    Load PET (+ geometry).
    Convert PET to SUV (if not provided).
    Align segmentations to PET.
    Crop organs from PET.
    Return dictionary of cropped PET organs (as np arrays).
    """

    import os
    import numpy as np
    import SimpleITK as sitk
    import pydicom
    import gc
    import math
    from pathlib import Path

    print(f"\nProcessing patient: {patient}")
    patient_path = os.path.join(data_dir, patient)

    if not os.path.isdir(patient_path):
        print("⚠️ Patient folder not found.")
        return None

    pet_folder = next(
        (os.path.join(patient_path, f) for f in os.listdir(patient_path)
         if 'PET' in f.upper() or 'FDG' in f.upper() or '5-MIN' in f.upper()),
        None
    )

    if pet_folder is None:
        print("⚠️ No PET folder found.")
        return None

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(pet_folder))
    reader.SetFileNames(dicom_names)
    pet_sitk = reader.Execute()

    # Enforce consistent orientation
    pet_sitk = sitk.DICOMOrient(pet_sitk, "LPS")


    dicom_files = list(Path(pet_folder).glob("*.dcm"))
    if len(dicom_files) == 0:
        print("⚠️ No PET DICOMs found.")
        return None

    ref_ds = pydicom.dcmread(dicom_files[0], force=True)

    # Extract patient physical attributes
    weight_kg = getattr(ref_ds, "PatientWeight", None)
    height_m = getattr(ref_ds, "PatientSize", None)

    if weight_kg is None or height_m is None:
        print("⚠️ Missing height or weight info.")
        bmi = None
    else:
        bmi = weight_kg / (height_m ** 2)

    weight_g = ref_ds.PatientWeight * 1000
    rad_info = ref_ds.RadiopharmaceuticalInformationSequence[0]
    injected_dose = rad_info.RadionuclideTotalDose
    half_life = rad_info.RadionuclideHalfLife

    def dicom_time_to_seconds(t):
        return int(t[:2])*3600 + int(t[2:4])*60 + float(t[4:])

    # Single delta_t for entire volume 
    delta_t = (
        dicom_time_to_seconds(ref_ds.AcquisitionTime)
        - dicom_time_to_seconds(rad_info.RadiopharmaceuticalStartTime)
    )
    if delta_t < 0:
        delta_t += 24 * 3600

    lambda_decay = math.log(2) / half_life
    decay_corrected_dose = injected_dose * math.exp(-lambda_decay * delta_t)

    # Get raw PET voxel data (SimpleITK handles orientation)
    pet_raw = sitk.GetArrayFromImage(pet_sitk).astype(np.float32)

    units_tag = (0x0054, 0x1001)
    units = str(ref_ds[units_tag].value) if units_tag in ref_ds else "Unknown"
    print(f"PET Units: {units}")

    if "BQML" in units.upper():
        activity_conc = pet_raw  # SKIP rescale - already Bq/ml!
        suv_factor = weight_g / decay_corrected_dose
        pet_np = activity_conc * suv_factor
    else:
        ref_slope = float(ref_ds.RescaleSlope) if 'RescaleSlope' in ref_ds else 1.0
        ref_intercept = float(ref_ds.RescaleIntercept) if 'RescaleIntercept' in ref_ds else 0.0
        activity_conc = pet_raw * ref_slope + ref_intercept
        pet_np = activity_conc * weight_g / decay_corrected_dose

    #Get the segmentations
    segmentation_folder = os.path.join(patient_path, 'CT_segmentation')

    if not os.path.exists(segmentation_folder):
        print("⚠️ No segmentation folder found.")
        return None

    available_segs = {}

    for seg_file in os.listdir(segmentation_folder):
        seg_path = os.path.join(segmentation_folder, seg_file)
        seg_img = sitk.ReadImage(seg_path)

        # Enforce same orientation before resampling
        seg_img = sitk.DICOMOrient(seg_img, "LPS")

        available_segs[seg_file] = seg_img

    # Apply combination logic
    if combination_logic is not None:
        for combined_name, components in combination_logic.items():
            combined_mask = None

            for comp in components:
                matched_file = next(
                    (f for f in available_segs if comp.lower() in f.lower()),
                    None
                )

                if matched_file:
                    part_mask = available_segs[matched_file]
                    combined_mask = part_mask if combined_mask is None else sitk.Or(combined_mask, part_mask)

            if combined_mask is not None:
                available_segs[combined_name] = combined_mask


    # Crop organs
    organs_to_process = list(organs_of_interest) + ['aorta']
    cropped_organs = {}

    # Get L1 range (for aorta restriction)
    l1_key = next((k for k in available_segs if "vertebrae_l1" in k.lower()), None)
    l1_z_range = None

    if l1_key is not None:
        aligned_l1 = sitk.Resample(
            available_segs[l1_key],
            pet_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )

        l1_np = sitk.GetArrayFromImage(aligned_l1)

        non_zero = np.argwhere(l1_np > 0)
        if non_zero.size > 0:
            min_z, _, _ = non_zero.min(axis=0)
            max_z, _, _ = non_zero.max(axis=0)
            l1_z_range = (min_z, max_z)

        del aligned_l1, l1_np
        gc.collect()

    # Process each organ
    for organ in organs_to_process:

        matched_key = next(
            (k for k in available_segs if organ.lower() in k.lower()),
            None
        )

        if matched_key is None:
            print(f"⚠️ {organ} not found.")
            continue

        seg_img = available_segs[matched_key]

        # Align segmentation to PET
        aligned_seg = sitk.Resample(
            seg_img,
            pet_sitk,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            sitk.sitkUInt8
        )

        seg_np = sitk.GetArrayFromImage(aligned_seg)

        # Check for mismatch
        if seg_np.shape != pet_np.shape:
            print(f"❌ Shape mismatch: PET {pet_np.shape} vs SEG {seg_np.shape}")
            continue

        # Aorta restriction
        if organ.lower() == "aorta" and l1_z_range is not None:
            zmin, zmax = l1_z_range
            seg_np[:zmin] = 0
            seg_np[zmax+1:] = 0

        non_zero = np.argwhere(seg_np > 0)
        if non_zero.size == 0:
            print(f"⚠️ {organ} mask empty.")
            continue

        min_z, min_y, min_x = non_zero.min(axis=0)
        max_z, max_y, max_x = non_zero.max(axis=0)

        # Apply mask before cropping
        masked_pet = np.zeros_like(pet_np)
        masked_pet[seg_np > 0] = pet_np[seg_np > 0]

        crop = masked_pet[min_z:max_z+1, min_y:max_y+1, min_x:max_x+1]

        cropped_organs[organ] = crop

        del aligned_seg, seg_np
        gc.collect()

    del pet_sitk
    gc.collect()

    return {
        "organs": cropped_organs,
        "patient_height_m": height_m,
        "patient_weight_kg": weight_kg,
        "patient_BMI": bmi
    }


from scipy.ndimage import binary_erosion
import numpy as np

def erode_organ_masks(organ_dict, erosion_config=None):
    """
    Apply per-organ boundary erosion to exclude segmentation edge contamination.
    
    Args:
        organ_dict: dict of {organ_name: cropped_pet_array}
        erosion_config: dict {organ: voxels_to_erode} or None for defaults
    
    Returns:
        eroded_organs: dict with eroded masks applied to PET values
    """
    
    eroded_organs = {}
    
    for organ, pet_crop in organ_dict.items():
        voxels_to_erode = erosion_config.get(organ, erosion_config['DEFAULT'])
        
        # Create mask from original 3D array
        binary_mask = pet_crop > 0
        
        orig_voxels = np.sum(binary_mask)
        if orig_voxels == 0:
            print(f"⚠️ {organ}: empty mask, skipping")
            eroded_organs[organ] = pet_crop
            continue
        
        # Erode the binary mask
        eroded_binary = binary_erosion(
            binary_mask, 
            structure=np.ones((3,3,3)),
            iterations=voxels_to_erode
        )
        
        eroded_crop = np.zeros_like(pet_crop)
        eroded_crop[eroded_binary] = pet_crop[eroded_binary]  
        
        new_voxels = np.sum(eroded_binary)
        reduction_pct = 100 * (1 - new_voxels / orig_voxels)
        
        eroded_organs[organ] = eroded_crop
        
        print(f"  {organ:<12} {orig_voxels:>6,} → {new_voxels:>6,} voxels "
              f"({reduction_pct:>4.1f}% eroded, {voxels_to_erode}v)")
    
    print(f"  Average reduction: {len(organ_dict)} organs processed\n")
    return eroded_organs