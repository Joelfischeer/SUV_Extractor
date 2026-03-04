
import os
import numpy as np
import SimpleITK as sitk
import pydicom
import gc
import math
from pathlib import Path

def PET_Organ_Cropper(
    data_dir,
    patient,
    organs_of_interest,
    combination_logic=None,
    custom_pet_array=None
):
    """
    Load CT and PET.
    Convert PET to SUV (if not provided).
    Align CT to PET.
    Apply combination logic to segmentations.
    Crop organs (organs_of_interest + aorta) from PET.
    Return dictionary of cropped PET organs (as np arrays).
    """

    print(f"\nProcessing patient: {patient}")
    patient_path = os.path.join(data_dir, patient)
    if not os.path.isdir(patient_path):
        print("⚠️ Patient folder not found.")
        return None

    # --------------------------------------------------
    # Load PET (or use precomputed SUV array)
    # --------------------------------------------------
    pet_np = custom_pet_array
    pet_sitk = None

    if pet_np is None:
        pet_folder = next(
            (os.path.join(patient_path, f) for f in os.listdir(patient_path)
             if 'PET' in f.upper() or 'FDG' in f.upper() or '5-MIN' in f.upper()),
            None
        )
        if pet_folder is None:
            print("⚠️ No PET folder found.")
            return None

        # Load PET DICOM series
        dicom_files = list(Path(pet_folder).glob("*.dcm"))
        if len(dicom_files) == 0:
            print("⚠️ No PET DICOMs found.")
            return None

        # Read metadata from first DICOM
        ds = pydicom.dcmread(str(dicom_files[0]))
        weight_g = ds.PatientWeight * 1000  # grams
        rad_info = ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = rad_info.RadionuclideTotalDose
        half_life = rad_info.RadionuclideHalfLife

        def dicom_time_to_seconds(t):
            return int(t[:2])*3600 + int(t[2:4])*60 + float(t[4:])

        delta_t = dicom_time_to_seconds(ds.AcquisitionTime) - dicom_time_to_seconds(rad_info.RadiopharmaceuticalStartTime)
        lambda_decay = math.log(2) / half_life
        decay_corrected_dose = injected_dose * math.exp(-lambda_decay * delta_t)

        # Load PET series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(str(pet_folder))
        reader.SetFileNames(dicom_names)
        pet_sitk = reader.Execute()
        pet_np = sitk.GetArrayFromImage(pet_sitk).astype(np.float32)

        # Convert to SUV
        pet_np = pet_np * weight_g / decay_corrected_dose

    # --------------------------------------------------
    # Load CT (only for alignment, optional)
    # --------------------------------------------------
    ct = None
    for f in os.listdir(patient_path):
        if 'CT' in f.upper() and 'SEGMENTATION' not in f.upper():
            ct_path = os.path.join(patient_path, f)
            if ct_path.endswith((".nii", ".nii.gz")):
                ct = sitk.ReadImage(ct_path)
            else:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
                reader.SetFileNames(dicom_names)
                ct = reader.Execute()
            break

    if ct is not None and pet_sitk is not None:
        aligned_ct = sitk.Resample(
            ct,
            pet_sitk,
            sitk.Transform(),
            sitk.sitkLinear,
            0.0,
            ct.GetPixelID()
        )
        del ct
        gc.collect()
    else:
        aligned_ct = None

    # --------------------------------------------------
    # Load segmentations
    # --------------------------------------------------
    segmentation_folder = os.path.join(patient_path, 'CT_segmentation')
    if not os.path.exists(segmentation_folder):
        print("⚠️ No segmentation folder found.")
        return None

    available_segs = {}
    for seg_file in os.listdir(segmentation_folder):
        seg_path = os.path.join(segmentation_folder, seg_file)
        seg_img = sitk.ReadImage(seg_path)
        available_segs[seg_file] = seg_img

    # --------------------------------------------------
    # Apply combination logic
    # --------------------------------------------------
    if combination_logic is not None:
        for combined_name, components in combination_logic.items():
            combined_mask = None
            for comp in components:
                matched_file = next(
                    (f for f in available_segs.keys() if comp.lower() in f.lower()),
                    None
                )
                if matched_file:
                    part_mask = available_segs[matched_file]
                    combined_mask = part_mask if combined_mask is None else sitk.Or(combined_mask, part_mask)
            if combined_mask is not None:
                available_segs[combined_name] = combined_mask

    # --------------------------------------------------
    # Crop organs from PET
    # --------------------------------------------------
    organs_to_process = list(organs_of_interest) + ['aorta']
    cropped_organs = {}

    for organ in organs_to_process:
        matched_key = next(
            (k for k in available_segs.keys() if organ.lower() in k.lower()),
            None
        )
        if matched_key is None:
            print(f"⚠️ {organ} not found.")
            continue

        seg_img = available_segs[matched_key]
        # Align segmentation → PET geometry
        if pet_sitk is not None:
            aligned_seg = sitk.Resample(
                seg_img,
                pet_sitk,
                sitk.Transform(),
                sitk.sitkNearestNeighbor,
                0,
                seg_img.GetPixelID()
            )
        else:
            aligned_seg = seg_img

        seg_np = sitk.GetArrayFromImage(aligned_seg)

        # Compute bounding box
        non_zero = np.argwhere(seg_np > 0)
        if non_zero.size == 0:
            print(f"⚠️ {organ} mask empty.")
            continue

        min_z, min_y, min_x = non_zero.min(axis=0)
        max_z, max_y, max_x = non_zero.max(axis=0)

        crop = pet_np[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]
        cropped_organs[organ] = crop

        del aligned_seg, seg_np
        gc.collect()

    if pet_sitk is not None:
        del pet_sitk
    if aligned_ct is not None:
        del aligned_ct
    gc.collect()

    return cropped_organs