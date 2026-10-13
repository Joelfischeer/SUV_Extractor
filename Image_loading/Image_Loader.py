import gc
import math
import os

import numpy as np

from .segmentation_sources import (seg_name as _seg_name,                 # noqa: F401 (kept for imports elsewhere)
                                   match_segmentation as _match_segmentation,  # noqa: F401
                                   get_segmentation_source, CombinedMasks)
from .folder_detection import find_pet_series


# SUV conversion - identical to Single_Patient_Image_Loader, so both give the
# same SUVs for the same patient.
def _dicom_time_to_seconds(t) -> float:
    """'101530.123' / '1015' -> seconds since midnight."""
    t = str(t).strip()
    hh = int(t[0:2]); mm = int(t[2:4]) if len(t) >= 4 else 0
    ss = float(t[4:]) if len(t) > 4 else 0.0
    return hh * 3600 + mm * 60 + ss


def _series_start_seconds(ref_ds, headers):
    """
    Start of the PET series in seconds since midnight: the SeriesTime, unless
    a slice was acquired earlier (then the earliest AcquisitionTime). A single
    slice's AcquisitionTime can be the time of a later bed position.
    """
    acq = [_dicom_time_to_seconds(h.AcquisitionTime) for h in headers
           if h.get("AcquisitionTime")]
    series = (_dicom_time_to_seconds(ref_ds.SeriesTime) if ref_ds.get("SeriesTime") else None)
    if series is not None and (not acq or series <= min(acq)):
        return series, "SeriesTime"
    if acq:
        return min(acq), "earliest AcquisitionTime"
    return series, "SeriesTime"


def _pet_to_suv(pet_raw, ref_ds, headers):
    """
    Convert an ALREADY RESCALED PET array to SUVbw.

      BQML  SUV = Bq/ml * weight [g] / dose at the decay-correction reference
              DecayCorrection 'ADMIN': images already corrected to the
                  injection time -> injected dose, no decay
              DecayCorrection 'START' (default) / other: dose decayed from
                  injection to the series start
      CNTS  Philips: SUV = rescaled value * Philips SUV scale factor (7053,1000)

    Returns (suv_array, info dict for auditing).
    """
    units = str(ref_ds.get("Units", "Unknown")).upper()
    info = {"units": units, "decay_correction": str(ref_ds.get("DecayCorrection", "")).upper()}

    if units == "BQML":
        weight_g = float(ref_ds.PatientWeight) * 1000
        rad_info = ref_ds.RadiopharmaceuticalInformationSequence[0]
        injected_dose = float(rad_info.RadionuclideTotalDose)
        half_life = float(rad_info.RadionuclideHalfLife)
        if info["decay_correction"] == "ADMIN":
            dose_at_ref, delta_t, ref_label = injected_dose, 0.0, "ADMIN (no decay)"
        else:
            start_s, ref_label = _series_start_seconds(ref_ds, headers)
            delta_t = start_s - _dicom_time_to_seconds(rad_info.RadiopharmaceuticalStartTime)
            if delta_t < 0:
                delta_t += 24 * 3600
            dose_at_ref = injected_dose * math.exp(-math.log(2) / half_life * delta_t)
        suv_factor = weight_g / dose_at_ref
        info.update({"method": f"BQML, decay to {ref_label}", "suv_factor": suv_factor,
                     "uptake_time_min": delta_t / 60, "injected_dose_Bq": injected_dose,
                     "weight_kg": weight_g / 1000})
        if not 20 <= delta_t / 60 <= 240 and info["decay_correction"] != "ADMIN":
            print(f"⚠️ Unusual uptake time {delta_t / 60:.0f} min - check the DICOM times.")
        return pet_raw * suv_factor, info

    if units == "CNTS":
        if (0x7053, 0x1000) not in ref_ds:
            raise ValueError("CNTS PET without Philips SUV scale factor (7053,1000).")
        philips_suv_factor = float(ref_ds[(0x7053, 0x1000)].value)
        if philips_suv_factor == 0:
            raise ValueError("Philips SUV Scale Factor is 0; CNTS cannot be converted to SUV.")
        info.update({"method": "CNTS, Philips SUV scale factor", "suv_factor": philips_suv_factor})
        return pet_raw * philips_suv_factor, info

    raise ValueError(f"Unsupported PET units: {units}")


def _restrict_to_z(mask, z_range):
    restricted = mask.copy()
    zmin, zmax = z_range
    restricted[:zmin] = 0
    restricted[zmax + 1:] = 0
    return restricted


def PET_Organ_Cropper(
    data_dir,
    patient,
    organs_of_interest,
    combination_logic=None,
    norm_reference="inferior_vena_cava",
    reference_vertebra="vertebrae_L2",
    segmentation_backend="totalsegmentator",
    pet_name_hints=None,
    moose_model_priority=None,
):
    """
    Load PET DICOM data for a patient, convert it to SUV units, align CT-based
    segmentations to the PET image space, and extract cropped PET volumes for
    specified organs, plus a blood-pool reference.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Root directory containing all patient subfolders.

    patient : str
        Patient identifier corresponding to a subfolder inside `data_dir`.

    organs_of_interest : list of str
        Organ names to extract (TotalSegmentator naming; MOOSE names are
        mapped to it, e.g. 'vertebra_L2' -> 'vertebrae_L2'). Names must match
        exactly (case-insensitive) or be defined in combination_logic; there
        is no partial/substring matching. Missing organs are skipped with a
        warning and give NaN in the results.

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

    segmentation_backend : 'totalsegmentator' or 'moose'
        'totalsegmentator': <patient>/CT_segmentation/<region>.nii.gz
        'moose': <patient>/moosez-*/segmentations/ (multi-label, newest run
                 per model)

    pet_name_hints : list of str, optional
        Preferred PET series names (folder or SeriesDescription), in priority
        order, e.g. ["PET_WB_CORRECTED"]. Only attenuation-corrected PT series
        in BQML/CNTS are considered in any case (see folder_detection).

    moose_model_priority : list of str, optional
        If a region is in several MOOSE models (e.g. 'aorta' in organs and
        cardiac), the model listed first wins.

    Returns
    -------
    dict or None
        - "organs" : dict[str, np.ndarray]
            Masked, tightly cropped PET volumes (SUV) per organ; the reference
            is included under its own name (e.g. 'inferior_vena_cava'),
            already restricted to the vertebra level.
        - "reference_organ" : str           name of the reference
        - "reference_SUV_mean" : float|None mean SUV of the restricted reference,
                                            eroded in-plane by one voxel (unless that
                                            would empty it) - divide organ SUVs by
                                            this for the normalised values
        - "reference_method" : str|None     e.g. 'inferior_vena_cava_L2_eroded';
                                            anything without '_L2' used the whole mask
        - "reference_voxels" : int          voxels the reference value is based on
        - "patient_height_m", "patient_weight_kg", "patient_BMI",
          "patient_time_of_scan"
        - "suv_conversion" : dict           units, decay correction, method, SUV
                                            factor and uptake time (for auditing)
        - "pet_series" : str                folder (relative to the patient) of the PET used
        - "segmentation_backend" : str
        - "organ_sources" : dict[str, str]  segmentation (and MOOSE model) per organ

        Returns None if required data (patient folder, PET data, or
        segmentations) are missing.

    Notes
    -----
    - All organ outputs are in SUV units (not normalised).
    - Segmentation alignment uses nearest-neighbour interpolation.
    """
    import SimpleITK as sitk
    import pydicom
    from scipy.ndimage import binary_erosion

    print(f"\nProcessing patient: {patient}")
    patient_path = os.path.join(data_dir, patient)

    if not os.path.isdir(patient_path):
        print("⚠️ Patient folder not found.")
        return None

    # ---------------- PET ----------------
    pet = find_pet_series(patient_path, name_hints=pet_name_hints)
    if pet is None:
        return None
    dicom_names = pet["files"]

    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    pet_sitk = reader.Execute()
    pet_sitk = sitk.DICOMOrient(pet_sitk, "LPS")

    # Headers of the series' own (sorted) files - also finds DICOMs without
    # a .dcm extension.
    headers = [pydicom.dcmread(f, stop_before_pixels=True, force=True) for f in dicom_names]
    ref_ds = headers[0]

    weight_kg = getattr(ref_ds, "PatientWeight", None)
    height_m = getattr(ref_ds, "PatientSize", None)
    time_scan = getattr(ref_ds, "SeriesTime", None)

    if not weight_kg or not height_m:
        print("⚠️ Missing height or weight info.")
        bmi = None
    else:
        bmi = float(weight_kg) / (float(height_m) ** 2)

    # SimpleITK applies RescaleSlope / RescaleIntercept per slice while
    # reading, so pet_raw is already in the scanner's unit (Bq/ml for BQML).
    # The slope must not be applied again.
    pet_raw = sitk.GetArrayFromImage(pet_sitk).astype(np.float32)
    pet_np, suv_info = _pet_to_suv(pet_raw, ref_ds, headers)
    del pet_raw
    print(f"PET Units: {suv_info['units']} | {suv_info['method']} | "
          f"factor {suv_info['suv_factor']:.4e}")

    # ---------------- Segmentations ----------------
    source = get_segmentation_source(patient_path, segmentation_backend, pet_sitk,
                                     moose_model_priority=moose_model_priority)
    if source is None:
        print("⚠️ No segmentation found.")
        return None
    masks = CombinedMasks(source, combination_logic)

    def _describe(key):
        if hasattr(source, "model_of") and key not in masks.combination_logic:
            return f"{key} [{source.model_of(key)}]"
        if key in masks.combination_logic:
            parts = [match_ for match_ in (_match_segmentation(source.names(), c)
                                           for c in masks.combination_logic[key]) if match_]
            return f"{key} = " + " | ".join(parts)
        return key

    # z-range of the reference vertebra in PET space
    vert_z_range = None
    vert_key = masks.resolve(reference_vertebra)
    if vert_key is not None:
        vert_np = masks.mask(vert_key)
        if vert_np is not None:
            non_zero = np.argwhere(vert_np > 0)
            if non_zero.size > 0:
                vert_z_range = (non_zero[:, 0].min(), non_zero[:, 0].max())
        del vert_np
    else:
        print(f"⚠️ {reference_vertebra} not found - reference not restricted.")

    # ---------------- Crop organs (+ the reference) ----------------
    organs_to_process = list(dict.fromkeys(list(organs_of_interest) + [norm_reference]))
    cropped_organs, organ_sources = {}, {}
    reference_value, reference_method, reference_voxels = None, None, 0
    level = reference_vertebra.split("_")[-1].upper()          # 'L2'

    for organ in organs_to_process:
        matched_key = masks.resolve(organ)
        if matched_key is None:
            print(f"⚠️ {organ} not found.")
            continue

        seg_np = masks.mask(matched_key)
        if seg_np is None:
            print(f"⚠️ {organ} not found.")
            continue

        if seg_np.shape != pet_np.shape:
            print(f"Shape mismatch: PET {pet_np.shape} vs SEG {seg_np.shape}")
            continue

        # Reference: restrict to the vertebra level, mean of the in-plane eroded mask
        if organ == norm_reference:
            method = "full"
            if vert_z_range is not None:
                restricted = _restrict_to_z(seg_np, vert_z_range)
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

        # Apply mask before cropping (crop first, so no full-size copy is needed)
        crop = (slice(min_z, max_z + 1), slice(min_y, max_y + 1), slice(min_x, max_x + 1))
        seg_crop = seg_np[crop] > 0
        cropped = np.zeros(seg_crop.shape, dtype=pet_np.dtype)
        cropped[seg_crop] = pet_np[crop][seg_crop]
        cropped_organs[organ] = cropped
        organ_sources[organ] = _describe(matched_key)

        del seg_np, seg_crop
        gc.collect()

    source.release()
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
        "suv_conversion": suv_info,
        "pet_series": os.path.relpath(pet["folder"], patient_path),
        "segmentation_backend": source.backend,
        "organ_sources": organ_sources,
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