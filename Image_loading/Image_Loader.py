def _seg_name(key):
    """'heart_atrium_left.nii.gz' -> 'heart_atrium_left' (lowercase, extension removed)."""
    k = str(key).lower()
    for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if k.endswith(ext):
            return k[: -len(ext)]
    return k


def _match_segmentation(keys, organ):
    """
    Exact file-name match first ('heart' -> 'heart.nii.gz'), substring match only
    as a fallback (e.g. combined masks). A substring-only match can return
    'heart_atrium_left.nii.gz' for 'heart', depending on the os.listdir order.
    """
    organ_l = organ.lower()
    exact = next((k for k in keys if _seg_name(k) == organ_l), None)
    if exact is not None:
        return exact
    return next((k for k in keys if organ_l in str(k).lower()), None)


def PET_Organ_Cropper(
    data_dir,
    patient,
    organs_of_interest,
    combination_logic=None,
    norm_reference="inferior_vena_cava",
    reference_vertebra="vertebrae_L2",
):
    """
    Load PET DICOM data for a patient, convert it to SUV units, align CT-based
    segmentations to the PET image space, and extract cropped PET volumes for
    specified organs, plus a blood-pool reference.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root directory containing all patient subfolders. Each patient folder
        must include:
        - A PET DICOM series (folder name containing "PET", "FDG", or "5-MIN")
        - A 'CT_segmentation' folder with organ segmentation masks

    patient : str
        Patient identifier corresponding to a subfolder inside `data_dir`.

    organs_of_interest : list of str
        Organ names to extract. Matched against the segmentation file names:
        exact name first ('heart' -> 'heart.nii.gz'), substring match only as a
        fallback. Missing or unmatched organs are skipped with a warning.

    combination_logic : dict, optional
        {"target_name": ["label1", "label2", ...]} - masks combined with OR.

    norm_reference : str
        Segmentation used as blood-pool reference, default the inferior vena
        cava. It is always added to the processed organs and restricted to the
        z-range of `reference_vertebra`.

    reference_vertebra : str
        Vertebra whose z-range restricts the reference (default L2). If the
        vertebra is missing or the restricted reference is empty, the whole
        reference mask is used and this is reported in `reference_method`.

    Returns
    -------
    dict or None
        - "organs" : dict[str, np.ndarray]
            Masked, tightly cropped PET volumes (SUV) per organ; the reference
            is included under its own name (e.g. 'inferior_vena_cava'),
            already restricted to the vertebra level.
        - "reference_organ" : str           name of the reference, e.g. 'inferior_vena_cava'
        - "reference_SUV_mean" : float|None mean SUV of the restricted reference,
                                            eroded in-plane by one voxel (unless that
                                            would empty it) - divide organ SUVs by
                                            this for the normalised values
        - "reference_method" : str|None     e.g. 'inferior_vena_cava_L2_eroded';
                                            anything without '_L2' used the whole mask
        - "reference_voxels" : int          voxels the reference value is based on
        - "patient_height_m", "patient_weight_kg", "patient_BMI",
          "patient_time_of_scan" : as before

        Returns None if required data (patient folder, PET data, or
        segmentations) are missing.

    Notes
    -----
    - All organ outputs are in SUV units (not normalised).
    - Segmentation alignment uses nearest-neighbour interpolation.
    - SUV conversion follows the main image loader: Bq/ml (BQML) with the
      decay-corrected dose, Philips counts (CNTS) with the Philips SUV scale
      factor (private tag 7053,1000).
    """
    import os
    import numpy as np
    import SimpleITK as sitk
    import pydicom
    import gc
    import math
    from pathlib import Path
    from scipy.ndimage import binary_erosion

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

    pet_sitk = sitk.DICOMOrient(pet_sitk, "LPS")

    # Header of one slice for the SUV information. Use the files the series
    # reader found, so DICOMs without a .dcm extension are found too.
    if len(dicom_names) == 0:
        print("⚠️ No PET DICOMs found.")
        return None
    ref_ds = pydicom.dcmread(dicom_names[0], force=True, stop_before_pixels=True)

    weight_kg = getattr(ref_ds, "PatientWeight", None)
    height_m = getattr(ref_ds, "PatientSize", None)
    time_scan = getattr(ref_ds, "SeriesTime", None)

    if not weight_kg or not height_m:
        print("⚠️ Missing height or weight info.")
        bmi = None
    else:
        bmi = float(weight_kg) / (float(height_m) ** 2)

    def dicom_time_to_seconds(t):
        t = str(t)
        return int(t[:2]) * 3600 + int(t[2:4]) * 60 + float(t[4:])

    pet_raw = sitk.GetArrayFromImage(pet_sitk).astype(np.float32)

    units = str(ref_ds.get("Units", "Unknown")).upper()
    print(f"PET Units: {units}")

    if "BQML" in units:
        # SimpleITK applies RescaleSlope / RescaleIntercept per slice when it
        # reads the series, so pet_raw is already in Bq/ml.
        weight_g = float(ref_ds.PatientWeight) * 1000
        rad_info = ref_ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = float(rad_info.RadionuclideTotalDose)
        half_life = float(rad_info.RadionuclideHalfLife)
        delta_t = (dicom_time_to_seconds(ref_ds.AcquisitionTime)
                   - dicom_time_to_seconds(rad_info.RadiopharmaceuticalStartTime))
        if delta_t < 0:
            delta_t += 24 * 3600
        decay_corrected_dose = injected_dose * math.exp(-math.log(2) / half_life * delta_t)
        pet_np = pet_raw * (weight_g / decay_corrected_dose)

    elif "CNTS" in units:
        # Philips: SUV = (rescaled value) * Philips SUV scale factor (7053,1000).
        # pet_raw is already rescaled by SimpleITK (see above).
        if (0x7053, 0x1000) not in ref_ds:
            raise ValueError("CNTS PET without Philips SUV scale factor (7053,1000).")
        philips_suv_factor = float(ref_ds[(0x7053, 0x1000)].value)
        if philips_suv_factor == 0:
            raise ValueError("Philips SUV Scale Factor is 0; CNTS cannot be converted to SUV.")
        pet_np = pet_raw * philips_suv_factor

    else:
        raise ValueError(f"Unsupported PET units: {units}")

    segmentation_folder = os.path.join(patient_path, 'CT_segmentation')

    if not os.path.exists(segmentation_folder):
        print("⚠️ No segmentation folder found.")
        return None

    available_segs = {}

    for seg_file in os.listdir(segmentation_folder):
        seg_path = os.path.join(segmentation_folder, seg_file)
        seg_img = sitk.ReadImage(seg_path)
        seg_img = sitk.DICOMOrient(seg_img, "LPS")
        available_segs[seg_file] = seg_img

    # Apply combination logic
    if combination_logic is not None:
        for combined_name, components in combination_logic.items():
            combined_mask = None
            for comp in components:
                matched_file = _match_segmentation(available_segs.keys(), comp)
                if matched_file:
                    part_mask = available_segs[matched_file]
                    combined_mask = part_mask if combined_mask is None else sitk.Or(combined_mask, part_mask)
            if combined_mask is not None:
                available_segs[combined_name] = combined_mask

    def _to_pet_space(seg_img):
        aligned = sitk.Resample(seg_img, pet_sitk, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
        return sitk.GetArrayFromImage(aligned)

    # z-range of the reference vertebra in PET space
    vert_key = _match_segmentation(available_segs.keys(), reference_vertebra)
    vert_z_range = None
    if vert_key is not None:
        vert_np = _to_pet_space(available_segs[vert_key])
        non_zero = np.argwhere(vert_np > 0)
        if non_zero.size > 0:
            vert_z_range = (non_zero[:, 0].min(), non_zero[:, 0].max())
        del vert_np
        gc.collect()
    else:
        print(f"⚠️ {reference_vertebra} not found - reference not restricted.")

    # Crop organs (+ the reference)
    organs_to_process = list(dict.fromkeys(list(organs_of_interest) + [norm_reference]))
    cropped_organs = {}
    reference_value, reference_method, reference_voxels = None, None, 0
    level = reference_vertebra.split("_")[-1].upper()          # 'L2'

    for organ in organs_to_process:
        matched_key = _match_segmentation(available_segs.keys(), organ)
        if matched_key is None:
            print(f"⚠️ {organ} not found.")
            continue

        seg_np = _to_pet_space(available_segs[matched_key])

        if seg_np.shape != pet_np.shape:
            print(f"Shape mismatch: PET {pet_np.shape} vs SEG {seg_np.shape}")
            continue

        # Reference: restrict to the vertebra level, mean of the in-plane eroded mask
        if organ == norm_reference:
            method = "full"
            if vert_z_range is not None:
                zmin, zmax = vert_z_range
                restricted = seg_np.copy()
                restricted[:zmin] = 0
                restricted[zmax + 1:] = 0
                if np.any(restricted > 0):
                    seg_np, method = restricted, level
                else:
                    print(f"⚠️ {organ} empty at the {level} level - using the whole mask.")
            ref_mask = seg_np > 0
            structure = np.zeros((3, 3, 3), dtype=bool)          # in-plane (y, x) erosion only
            structure[1] = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
            eroded = binary_erosion(ref_mask, structure=structure)
            if eroded.any():
                ref_mask, method = eroded, method + "_eroded"
            if ref_mask.any():
                reference_value = float(pet_np[ref_mask].mean())
                reference_method = f"{organ}_{method}"
                reference_voxels = int(ref_mask.sum())
            if not method.startswith(level):
                print(f"⚠️ {organ}: reference not restricted to {level} ({method}).")

        non_zero = np.argwhere(seg_np > 0)
        if non_zero.size == 0:
            print(f"⚠️ {organ} mask empty.")
            continue

        min_z, min_y, min_x = non_zero.min(axis=0)
        max_z, max_y, max_x = non_zero.max(axis=0)

        # Apply mask before cropping
        masked_pet = np.zeros_like(pet_np)
        masked_pet[seg_np > 0] = pet_np[seg_np > 0]

        cropped_organs[organ] = masked_pet[min_z:max_z + 1, min_y:max_y + 1, min_x:max_x + 1]

        del seg_np, masked_pet
        gc.collect()

    del pet_sitk
    gc.collect()

    if reference_value is None:
        print(f"⚠️ No valid {norm_reference} reference value.")

    return {
        "organs": cropped_organs,
        "reference_organ": norm_reference,
        "reference_SUV_mean": reference_value,
        "reference_method": reference_method,
        "reference_voxels": reference_voxels,
        "patient_height_m": height_m,
        "patient_weight_kg": weight_kg,
        "patient_BMI": bmi,
        "patient_time_of_scan": time_scan,
    }


def erode_organ_masks(organ_dict, erosion_config=None, min_voxels=10):
    """
    Apply per-organ boundary erosion to exclude segmentation edge contamination.

    Args:
        organ_dict: dict of {organ_name: cropped_pet_array}
        erosion_config: dict {organ: voxels_to_erode, 'DEFAULT': voxels} or
            None (= 1 voxel for every organ)
        min_voxels: if erosion would leave fewer voxels than this (e.g. a thin
            vessel segment), the organ is kept uneroded and a warning printed.

    Returns:
        eroded_organs: dict with eroded masks applied to PET values
    """
    from scipy.ndimage import binary_erosion
    import numpy as np

    erosion_config = {"DEFAULT": 1, **(erosion_config or {})}
    eroded_organs = {}

    for organ, pet_crop in organ_dict.items():
        voxels_to_erode = erosion_config.get(organ, erosion_config["DEFAULT"])

        binary_mask = pet_crop > 0
        orig_voxels = int(np.sum(binary_mask))
        if orig_voxels == 0:
            print(f"⚠️ {organ}: empty mask, skipping")
            eroded_organs[organ] = pet_crop
            continue
        if voxels_to_erode <= 0:
            eroded_organs[organ] = pet_crop
            continue

        eroded_binary = binary_erosion(binary_mask, structure=np.ones((3, 3, 3)),
                                       iterations=voxels_to_erode)
        new_voxels = int(np.sum(eroded_binary))
        if new_voxels < min_voxels:
            print(f"⚠️ {organ}: erosion by {voxels_to_erode}v would leave {new_voxels} voxels "
                  f"- kept uneroded")
            eroded_organs[organ] = pet_crop
            continue

        eroded_crop = np.zeros_like(pet_crop)
        eroded_crop[eroded_binary] = pet_crop[eroded_binary]
        eroded_organs[organ] = eroded_crop

        print(f"  {organ:<20} {orig_voxels:>6,} → {new_voxels:>6,} voxels "
              f"({100 * (1 - new_voxels / orig_voxels):>4.1f}% eroded, {voxels_to_erode}v)")

    print(f"  {len(organ_dict)} organs processed\n")
    return eroded_organs