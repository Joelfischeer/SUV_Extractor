import gc
import math
import os
from pathlib import Path

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


# ---------------------------------------------------------------------------
# PET loading: DICOM, with NIfTI fallback for implausible SUVs
# ---------------------------------------------------------------------------
# Decisions per patient within a run, so every extraction (both tools, the
# comparison) uses the same PET source for a patient.
_PET_SOURCE_DECISIONS = {}


def reset_pet_source_decisions():
    _PET_SOURCE_DECISIONS.clear()


def pet_source_decision(patient_path):
    """NIfTI fallback decided for this patient in this run (dict) or None."""
    return _PET_SOURCE_DECISIONS.get(str(Path(patient_path).resolve()))


def _read_dicom_pet(pet):
    """DICOM series -> (pet_sitk LPS, SUV array, headers, suv_info). Raises on failure."""
    import SimpleITK as sitk
    import pydicom

    dicom_names = pet["files"]
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(dicom_names)
    pet_sitk = sitk.DICOMOrient(reader.Execute(), "LPS")
    headers = [pydicom.dcmread(f, stop_before_pixels=True, force=True) for f in dicom_names]
    # SimpleITK applies RescaleSlope / RescaleIntercept per slice while
    # reading, so the array is already in the scanner's unit (Bq/ml for BQML).
    pet_raw = sitk.GetArrayFromImage(pet_sitk).astype(np.float32)
    pet_np, suv_info = _pet_to_suv(pet_raw, headers[0], headers)
    return pet_sitk, pet_np, headers, suv_info


def _read_nifti_pet(path):
    """NIfTI -> (pet_sitk LPS, raw array) - unit unknown, decided by plausibility."""
    import SimpleITK as sitk
    img = sitk.ReadImage(str(path))
    if img.GetDimension() != 3:
        raise ValueError(f"PET NIfTI has {img.GetDimension()} dimensions, expected 3")
    img = sitk.DICOMOrient(img, "LPS")
    return img, sitk.GetArrayFromImage(img).astype(np.float32)


def _organ_means(cropped):
    out = {}
    for organ, arr in cropped.items():
        vals = arr[arr > 0]
        if vals.size:
            out[organ] = float(vals.mean())
    return out


def _plausible(means, factor, threshold):
    """All organ SUV_means <= threshold and the median organ in a physiological range."""
    vals = np.array(list(means.values())) * factor
    return bool(len(vals)) and vals.max() <= threshold and 0.2 <= np.median(vals) <= 15


def _crop_organs(pet_sitk, pet_np, patient_path, segmentation_backend, organs_to_process,
                 combination_logic, norm_reference, reference_vertebra, moose_model_priority):
    """
    Resample the segmentations to the PET grid and crop every organ (+ the
    reference). Returns a dict or None if there is no segmentation.
    """
    from scipy.ndimage import binary_erosion

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
            parts = [m for m in (_match_segmentation(source.names(), c)
                                 for c in masks.combination_logic[key]) if m]
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
    gc.collect()
    return {"organs": cropped_organs, "organ_sources": organ_sources,
            "reference_SUV_mean": reference_value, "reference_method": reference_method,
            "reference_voxels": reference_voxels, "segmentation_backend": source.backend}


def _scale(crop_result, factor):
    out = dict(crop_result)
    out["organs"] = {o: a * np.float32(factor) for o, a in crop_result["organs"].items()}
    if out["reference_SUV_mean"] is not None:
        out["reference_SUV_mean"] = out["reference_SUV_mean"] * factor
    return out


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
    suv_outlier_threshold=20.0,
    pet_nifti_hints=None,
):
    """
    Load PET data for a patient, convert it to SUV units, align CT-based
    segmentations to the PET image space, and extract cropped PET volumes for
    specified organs, plus a blood-pool reference.

    Parameters
    ----------
    data_dir, patient : cohort folder and patient sub-folder.
    organs_of_interest : list of str
        Organ names (TotalSegmentator naming; MOOSE names are mapped to it,
        e.g. 'vertebra_L2' -> 'vertebrae_L2'). Names must match exactly
        (case-insensitive) or be defined in combination_logic. Missing organs
        are skipped with a warning and give NaN in the results.
    combination_logic : dict, optional
        {"target_name": ["label1", "label2", ...]} - masks combined with OR.
    norm_reference, reference_vertebra : blood-pool reference (IVC) restricted
        to the z-range of the vertebra (L2).
    segmentation_backend : 'totalsegmentator' or 'moose'
    pet_name_hints : preferred PET DICOM series names.
    moose_model_priority : MOOSE sub-model order for regions in several models.
    suv_outlier_threshold : float
        If any organ SUV_mean exceeds this with the DICOM PET (or the DICOM PET
        cannot be read), the issue is logged and the PET is loaded from the
        patient's PET NIfTI instead (e.g. wb_pet.nii.gz). Its unit is decided
        by plausibility: scanner units x DICOM SUV factor, or already SUV.
        If the NIfTI does not help, the DICOM values are kept and flagged.
    pet_nifti_hints : preferred PET NIfTI names (default ['WB_PET', 'PET']).

    Returns
    -------
    dict or None
        "organs", "reference_organ", "reference_SUV_mean", "reference_method",
        "reference_voxels", "patient_height_m", "patient_weight_kg",
        "patient_BMI", "patient_time_of_scan", "suv_conversion",
        "pet_series", "pet_source" ('dicom' or 'nifti: <file> (<unit>)'),
        "pet_issue" ('' or a description), "segmentation_backend",
        "organ_sources".
        None if PET or segmentations are missing.
    """
    from .issue_log import log_issue
    from .folder_detection import find_pet_nifti

    print(f"\nProcessing patient: {patient}")
    patient_path = os.path.join(data_dir, patient)

    if not os.path.isdir(patient_path):
        print("⚠️ Patient folder not found.")
        return None

    organs_to_process = list(dict.fromkeys(list(organs_of_interest) + [norm_reference]))
    crop_args = (patient_path, segmentation_backend, organs_to_process, combination_logic,
                 norm_reference, reference_vertebra, moose_model_priority)

    # ---------------- PET: DICOM ----------------
    pet = find_pet_series(patient_path, name_hints=pet_name_hints)
    headers, suv_info, dicom_error = None, {}, None
    pet_sitk = pet_np = None
    if pet is not None:
        try:
            pet_sitk, pet_np, headers, suv_info = _read_dicom_pet(pet)
            print(f"PET Units: {suv_info['units']} | {suv_info['method']} | "
                  f"factor {suv_info['suv_factor']:.4e}")
        except Exception as e:
            dicom_error = f"PET DICOM could not be read/converted ({e})"
            if headers is None:
                try:     # headers alone (patient data, SUV factor) may still be readable
                    import pydicom
                    headers = [pydicom.dcmread(f, stop_before_pixels=True, force=True)
                               for f in pet["files"]]
                    suv_info = _pet_to_suv(np.zeros(1, np.float32), headers[0], headers)[1]
                except Exception:
                    pass
    else:
        dicom_error = "no usable PET DICOM series"

    decision = pet_source_decision(patient_path)
    result, pet_source, pet_issue = None, "dicom", ""

    if dicom_error is None and decision is None:
        result = _crop_organs(pet_sitk, pet_np, *crop_args)
        if result is None:
            return None
        means = _organ_means(result["organs"])
        outliers = {o: m for o, m in means.items() if m > suv_outlier_threshold}
        if outliers:
            listing = ", ".join(f"{o} {m:.1f}" for o, m in sorted(outliers.items(), key=lambda x: -x[1]))
            pet_issue = f"SUV_mean > {suv_outlier_threshold:g} with DICOM PET: {listing}"
    elif dicom_error is not None:
        pet_issue = dicom_error
    del pet_np
    gc.collect()

    # ---------------- PET: NIfTI fallback ----------------
    if pet_issue or decision is not None:
        if decision is None:
            nifti = find_pet_nifti(patient_path, pet_nifti_hints)
        else:
            nifti = Path(decision["path"])
        rel = os.path.relpath(nifti, patient_path) if nifti else None

        if nifti is None:
            if pet_issue:
                log_issue(patient, "PET", f"{pet_issue} -> no PET NIfTI found, "
                                         f"{'DICOM values kept - check manually' if result else 'no result'}")
        else:
            try:
                nifti_sitk, nifti_raw = _read_nifti_pet(nifti)
                raw_result = _crop_organs(nifti_sitk, nifti_raw, *crop_args)
                del nifti_raw
                gc.collect()
                if raw_result is None:
                    return None
                if decision is not None:
                    factor, unit = decision["factor"], decision["unit"]
                else:
                    raw_means = _organ_means(raw_result["organs"])
                    options = []
                    if suv_info.get("suv_factor"):
                        options.append(("scanner units x DICOM SUV factor", suv_info["suv_factor"]))
                    options.append(("already SUV", 1.0))
                    if "SUV" in str(nifti.name).upper():
                        options.reverse()
                    plausible = [o for o in options
                                 if _plausible(raw_means, o[1], suv_outlier_threshold)]
                    unit, factor = plausible[0] if plausible else (None, None)
                    if factor is None:
                        log_issue(patient, "PET", f"{pet_issue} -> NIfTI {rel} gives no plausible "
                                                  f"SUVs either (raw organ means "
                                                  f"{', '.join(f'{o} {m:.3g}' for o, m in list(raw_means.items())[:4])}) "
                                                  f"-> {'DICOM values kept - check segmentation/CT' if result else 'no result'}")
                    else:
                        _PET_SOURCE_DECISIONS[str(Path(patient_path).resolve())] = {
                            "path": str(nifti), "factor": factor, "unit": unit, "issue": pet_issue}
                        new_max = max(_organ_means(_scale(raw_result, factor)["organs"]).values())
                        log_issue(patient, "PET", f"{pet_issue} -> using NIfTI {rel} ({unit}); "
                                                  f"highest organ SUV_mean now {new_max:.2f}")
                if factor is not None:
                    result = _scale(raw_result, factor)
                    pet_source = f"nifti: {rel} ({unit})"
                    if decision is not None:
                        pet_issue = decision["issue"]
                    if decision is None and suv_info:
                        suv_info = dict(suv_info, method=f"{suv_info.get('method', '')} -> NIfTI {unit}")
            except Exception as e:
                log_issue(patient, "PET", f"{pet_issue} -> NIfTI {rel} failed ({e})")

    if result is None:
        return None

    ref_ds = headers[0] if headers else {}
    weight_kg = ref_ds.get("PatientWeight", None) if headers else None
    height_m = ref_ds.get("PatientSize", None) if headers else None
    time_scan = ref_ds.get("SeriesTime", None) if headers else None
    if not weight_kg or not height_m:
        print("⚠️ Missing height or weight info.")
        bmi = None
    else:
        bmi = float(weight_kg) / (float(height_m) ** 2)

    if result["reference_SUV_mean"] is None:
        print(f"⚠️ No valid {norm_reference} reference value.")

    return {
        "organs": result["organs"],
        "reference_organ": norm_reference,
        "reference_SUV_mean": result["reference_SUV_mean"],
        "reference_method": result["reference_method"],
        "reference_voxels": result["reference_voxels"],
        "patient_height_m": height_m,
        "patient_weight_kg": weight_kg,
        "patient_BMI": bmi,
        "patient_time_of_scan": time_scan,
        "suv_conversion": suv_info,
        "pet_series": os.path.relpath(pet["folder"], patient_path) if pet else None,
        "pet_source": pet_source,
        "pet_issue": pet_issue,
        "segmentation_backend": result["segmentation_backend"],
        "organ_sources": result["organ_sources"],
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