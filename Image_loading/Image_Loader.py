import os
import numpy as np
import SimpleITK as sitk
import gc


def PET_Organ_Cropper(
    data_dir,
    patient,
    organs_of_interest,
    combination_logic=None
):
    """
    Load CT and PET.
    Align CT to PET.
    Apply combination logic to segmentations.
    Crop organs (organs_of_interest + aorta) from PET.
    Return dictionary of cropped PET organs.
    """

    print(f"\nProcessing patient: {patient}")

    patient_path = os.path.join(data_dir, patient)
    if not os.path.isdir(patient_path):
        print("⚠️ Patient folder not found.")
        return None

    image_files = os.listdir(patient_path)

    # --------------------------------------------------
    # Load CT
    # --------------------------------------------------
    ct = None
    for image_file in image_files:
        if 'CT' in image_file.upper() and 'SEGMENTATION' not in image_file.upper():
            ct_path = os.path.join(patient_path, image_file)

            if ct_path.endswith((".nii", ".nii.gz")):
                ct = sitk.ReadImage(ct_path)
            else:
                reader = sitk.ImageSeriesReader()
                dicom_names = reader.GetGDCMSeriesFileNames(ct_path)
                reader.SetFileNames(dicom_names)
                ct = reader.Execute()
            break

    if ct is None:
        print("⚠️ No CT found.")
        return None

    # --------------------------------------------------
    # Load PET
    # --------------------------------------------------
    pet = None
    for image_file in image_files:
        name_upper = image_file.upper()
        if 'PET' in name_upper or 'FDG' in name_upper or '5-MIN' in name_upper:
            pet_path = os.path.join(patient_path, image_file)

            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(pet_path)
            reader.SetFileNames(dicom_names)
            pet = reader.Execute()
            break

    if pet is None:
        print("⚠️ No PET found.")
        return None

    # --------------------------------------------------
    # Align CT → PET (geometry only)
    # --------------------------------------------------
    aligned_ct = sitk.Resample(
        ct,
        pet,
        sitk.Transform(),
        sitk.sitkLinear,
        0.0,
        ct.GetPixelID()
    )

    del ct
    gc.collect()

    pet_np = sitk.GetArrayFromImage(pet)

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
                    (f for f in available_segs.keys()
                     if comp.lower() in f.lower()),
                    None
                )

                if matched_file:
                    part_mask = available_segs[matched_file]
                    combined_mask = (
                        part_mask if combined_mask is None
                        else sitk.Or(combined_mask, part_mask)
                    )

            if combined_mask is not None:
                available_segs[combined_name] = combined_mask

    # --------------------------------------------------
    # Crop organs from PET
    # --------------------------------------------------
    organs_to_process = list(organs_of_interest) + ['aorta']
    cropped_organs = {}

    for organ in organs_to_process:

        matched_key = next(
            (k for k in available_segs.keys()
             if organ.lower() in k.lower()),
            None
        )

        if matched_key is None:
            print(f"⚠️ {organ} not found.")
            continue

        seg_img = available_segs[matched_key]

        # Align segmentation → PET
        aligned_seg = sitk.Resample(
            seg_img,
            pet,
            sitk.Transform(),
            sitk.sitkNearestNeighbor,
            0,
            seg_img.GetPixelID()
        )

        seg_np = sitk.GetArrayFromImage(aligned_seg)

        # Compute bounding box
        non_zero = np.argwhere(seg_np > 0)
        if non_zero.size == 0:
            print(f"⚠️ {organ} mask empty.")
            continue

        min_z, min_y, min_x = non_zero.min(axis=0)
        max_z, max_y, max_x = non_zero.max(axis=0)

        crop = pet_np[
            min_z:max_z + 1,
            min_y:max_y + 1,
            min_x:max_x + 1
        ]

        cropped_organs[organ] = crop

        print(f"✓ Cropped {organ}: {crop.shape}")

        del aligned_seg, seg_np
        gc.collect()

    del pet, aligned_ct
    gc.collect()

    return cropped_organs