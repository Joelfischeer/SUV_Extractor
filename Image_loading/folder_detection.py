"""
Robust detection of the PET series and the CT image inside a patient folder.

PET:  only DICOM series with Modality 'PT' are considered (SUV conversion needs
      the DICOM headers). Non-attenuation-corrected series, MIPs/screenshots
      and series in other units than BQML/CNTS are rejected. The remaining
      series are ranked by the name hints (folder name or SeriesDescription),
      then by number of slices (whole body wins).

CT:   NIfTI files whose name contains a 'CT' token (AC_CT.nii.gz, CT_xxx.nii)
      and DICOM series with Modality 'CT'. Ranked by the name hints, then
      aligned '*_corrected.nii.gz' versions first, NIfTI before DICOM, then
      the largest image.

Name matching is token-based ('PET_WB_CORRECTED' -> PET, WB, CORRECTED), so
'CORRECTED' is never mistaken for a CT and 'UNCORRECTED' never for 'CORRECTED'.
Segmentation outputs (CT_segmentation, moosez-*) are never searched.
"""

import os
import re
from pathlib import Path

# Folders that never contain the source images
SKIP_DIR_PATTERNS = (
    re.compile(r"^moosez-", re.I),
    re.compile(r"segmentation", re.I),
    re.compile(r"^\."),
    re.compile(r"^__"),
)

NIFTI_EXT = (".nii.gz", ".nii")

# Tokens marking a NIfTI that is not a CT image (masks, labels, derived files)
NON_IMAGE_TOKENS = {"SEG", "SEGMENTATION", "SEGMENTATIONS", "MASK", "MASKS",
                    "LABEL", "LABELS", "ORGANS", "BODY"}

# Tokens marking a PET series that must not be used
PET_REJECT_TOKENS = {"NAC", "NOAC", "NONAC", "UNCORRECTED", "NONCORRECTED",
                     "MIP", "MIPS", "SCREEN", "SCREENSHOT", "CAPTURE", "RAW"}


def _tokens(text):
    return set(t for t in re.split(r"[^A-Z0-9]+", str(text).upper()) if t)


def _hint_rank(texts, hints):
    """Index of the first hint found in any of the texts (len(hints) if none).
    A hint matches as a substring of the upper-cased text."""
    up = [str(t).upper() for t in texts if t]
    for i, hint in enumerate(hints or []):
        h = hint.upper()
        if any(h in t for t in up):
            return i
    return len(hints or [])


def _skip_dir(name):
    return any(p.search(name) for p in SKIP_DIR_PATTERNS)


def _walk(root, max_depth):
    """os.walk restricted to max_depth, skipping segmentation folders."""
    root = Path(root)
    base_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).parts) - base_depth
        dirnames[:] = sorted(d for d in dirnames if not _skip_dir(d))
        if depth >= max_depth:
            dirnames[:] = []
        yield Path(dirpath), dirnames, sorted(filenames)


def _looks_like_dicom(path):
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _dir_has_dicom(dirpath, filenames, n_check=5):
    candidates = [f for f in filenames if not f.lower().endswith(NIFTI_EXT + (".json", ".csv", ".txt"))]
    return any(_looks_like_dicom(dirpath / f) for f in candidates[:n_check])


def list_dicom_series(patient_path, max_depth=4):
    """
    All DICOM series below patient_path.

    Returns a list of dicts: folder, series_id, files (sorted by GDCM),
    header (pydicom dataset of the first file, no pixels).
    """
    import SimpleITK as sitk
    import pydicom

    series = []
    for dirpath, _, filenames in _walk(patient_path, max_depth):
        if not filenames or not _dir_has_dicom(dirpath, filenames):
            continue
        try:
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(str(dirpath))
        except RuntimeError:
            continue
        for sid in series_ids:
            files = list(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(str(dirpath), sid))
            if not files:
                continue
            try:
                header = pydicom.dcmread(files[0], stop_before_pixels=True, force=True)
            except Exception:
                continue
            series.append({"folder": dirpath, "series_id": sid, "files": files, "header": header})
    return series


def find_pet_series(patient_path, name_hints=None, verbose=True):
    """
    Select the attenuation-corrected whole-body PET series.

    Parameters
    ----------
    patient_path : str or Path
    name_hints : list of str, optional
        Preferred names in priority order, matched against the folder name and
        SeriesDescription, e.g. ["PET_WB_CORRECTED", "LAST-5-MIN"].

    Returns
    -------
    dict (folder, series_id, files, header) or None
    """
    candidates, rejected = [], []

    for s in list_dicom_series(patient_path):
        h = s["header"]
        name = s["folder"].name
        desc = str(h.get("SeriesDescription", ""))
        label = f"{name} / '{desc}'"

        if str(h.get("Modality", "")).upper() != "PT":
            continue

        tokens = _tokens(name) | _tokens(desc)
        bad = tokens & PET_REJECT_TOKENS
        if bad:
            rejected.append((label, f"name contains {sorted(bad)}"))
            continue

        corrected = h.get("CorrectedImage", None)      # e.g. ['DECY', 'ATTN', 'SCAT', ...]
        if corrected is not None:
            values = [corrected] if isinstance(corrected, str) else list(corrected)
            if "ATTN" not in [str(v).upper() for v in values]:
                rejected.append((label, "not attenuation corrected (no ATTN in CorrectedImage)"))
                continue

        units = str(h.get("Units", "")).upper()
        if units not in ("BQML", "CNTS"):
            rejected.append((label, f"units '{units}' (need BQML or CNTS)"))
            continue

        image_type = [str(x).upper() for x in (h.get("ImageType", []) or [])]
        if "SECONDARY" in image_type and len(s["files"]) <= 3:
            rejected.append((label, "secondary capture"))
            continue

        rank = _hint_rank([name, desc], name_hints)
        candidates.append(((rank, -len(s["files"])), s, label))

    if not candidates:
        if verbose:
            print("⚠️ No usable PET series found.")
            for label, why in rejected:
                print(f"     rejected {label}: {why}")
        return None

    candidates.sort(key=lambda c: c[0])
    _, best, label = candidates[0]
    if verbose:
        print(f"PET series: {label} ({len(best['files'])} slices)")
        others = [c[2] for c in candidates[1:]]
        if others:
            print(f"     other usable PET series ignored: {others}")
        for lab, why in rejected:
            print(f"     rejected {lab}: {why}")
    return best


def _nifti_size(path):
    try:
        return os.path.getsize(path)
    except OSError:
        return 0


def list_ct_sources(patient_path, name_hints=None, max_depth=4):
    """
    All CT candidates of a patient, best first.

    Ranking: name hint, aligned '*_corrected.nii.gz' first, NIfTI before DICOM,
    then the largest image. Each entry: dict with 'path' (Path: NIfTI file or
    DICOM folder), 'kind' ('nifti' or 'dicom'), 'series_id' (DICOM only).
    """
    candidates = []

    for dirpath, _, filenames in _walk(patient_path, max_depth):
        for f in filenames:
            if not f.lower().endswith(NIFTI_EXT):
                continue
            stem = re.sub(r"(\.nii\.gz|\.nii)$", "", f, flags=re.I)
            tokens = _tokens(stem)
            if "CT" not in tokens or tokens & NON_IMAGE_TOKENS or tokens & {"PET", "PT", "SUV"}:
                continue
            p = dirpath / f
            corrected = f.lower().endswith("corrected.nii.gz")
            rank = _hint_rank([f, dirpath.name], name_hints)
            candidates.append(((rank, not corrected, 0, -_nifti_size(p)),
                               {"path": p, "kind": "nifti", "series_id": None}))

    for s in list_dicom_series(patient_path, max_depth):
        h = s["header"]
        if str(h.get("Modality", "")).upper() != "CT":
            continue
        image_type = [str(x).upper() for x in (h.get("ImageType", []) or [])]
        if "LOCALIZER" in image_type or len(s["files"]) < 20:
            continue  # scout / topogram
        rank = _hint_rank([s["folder"].name, h.get("SeriesDescription", "")], name_hints)
        candidates.append(((rank, True, 1, -len(s["files"])),
                           {"path": s["folder"], "kind": "dicom", "series_id": s["series_id"]}))

    candidates.sort(key=lambda c: c[0])
    return [c[1] for c in candidates]


def find_ct_source(patient_path, name_hints=None, max_depth=4, verbose=True):
    """
    Select the CT image of a patient (best of list_ct_sources).

    Returns dict with keys 'path', 'kind' ('nifti' or 'dicom'), 'series_id'
    (DICOM only), or None.
    """
    candidates = list_ct_sources(patient_path, name_hints, max_depth)
    if not candidates:
        if verbose:
            print(f"⚠️ No CT found in {patient_path}")
        return None
    best = candidates[0]
    if verbose:
        rel = os.path.relpath(best["path"], patient_path)
        print(f"CT source: {rel} ({best['kind']})")
    return best


def find_pet_nifti(patient_path, name_hints=None, max_depth=4):
    """
    PET NIfTI of a patient (fallback when the PET DICOMs are faulty): a
    .nii/.nii.gz with a 'PET' token in its name, e.g. wb_pet.nii.gz, PET.nii.
    Masks, NAC, MIP and CT files are excluded. Ranked by name hints
    (default: 'WB_PET' before plain 'PET'), then the largest file.
    Returns a Path or None.
    """
    hints = name_hints or ["WB_PET", "PET"]
    candidates = []
    for dirpath, _, filenames in _walk(patient_path, max_depth):
        for f in filenames:
            if not f.lower().endswith(NIFTI_EXT):
                continue
            stem = re.sub(r"(\.nii\.gz|\.nii)$", "", f, flags=re.I)
            tokens = _tokens(stem)
            if "PET" not in tokens and "WBPET" not in tokens:
                continue
            if tokens & (NON_IMAGE_TOKENS | PET_REJECT_TOKENS | {"CT"}):
                continue
            p = dirpath / f
            candidates.append(((_hint_rank([f], hints), -_nifti_size(p)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    return candidates[0][1]


def read_ct_image(ct_source):
    """Read the CT selected by find_ct_source as a SimpleITK image."""
    import SimpleITK as sitk
    if ct_source["kind"] == "nifti":
        return sitk.ReadImage(str(ct_source["path"]))
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames(sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
        str(ct_source["path"]), ct_source["series_id"]))
    return reader.Execute()


def ct_candidates_for_segmentation(patient_path, name_hints=None):
    """
    CT images to try for a segmentation, in order: the best candidate, then
    the other NIfTI files with 'CT' in the name, then other CT DICOM series.
    Used to retry when a segmentation fails, e.g. because the CT DICOMs of the
    AC_CT folder are faulty.
    """
    ranked = list_ct_sources(patient_path, name_hints)
    if not ranked:
        return []
    rest = ranked[1:]
    return [ranked[0]] + [c for c in rest if c["kind"] == "nifti"] + [c for c in rest if c["kind"] == "dicom"]


def check_ct_image(ct_source):
    """
    Is the CT usable for a segmentation? Faulty DICOMs often load without an
    error but give a garbage image, so the content is checked:
    >= 20 slices, air present (1st percentile < -500 HU, every CT has air
    around the body), tissue present (99th percentile > -200 HU), contrast.
    Returns (ok, reason).
    """
    import numpy as np
    import SimpleITK as sitk
    try:
        img = read_ct_image(ct_source)
    except Exception as e:
        return False, f"cannot be read ({e})"
    if img.GetDimension() != 3:
        return False, f"{img.GetDimension()}D image"
    arr = sitk.GetArrayFromImage(img)
    if arr.shape[0] < 20:
        return False, f"only {arr.shape[0]} slices"
    sample = arr[:, ::4, ::4].astype(np.float32)
    p1, p99, sd = np.percentile(sample, 1), np.percentile(sample, 99), sample.std()
    if p1 > -500 or p99 < -200 or sd < 20:
        return False, f"implausible CT values (1st/99th percentile {p1:.0f}/{p99:.0f} HU, SD {sd:.0f})"
    return True, ""