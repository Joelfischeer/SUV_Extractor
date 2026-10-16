"""
TotalSegmentator per patient and task.

Output: <patient>/CT_segmentation/<region>.nii.gz (one file per region; every
task writes into the same folder).

A task counts as done if
  - its completion stamp CT_segmentation/.totalseg_<task>.done exists (written
    after every successful run here), or
  - its marker files exist (for segmentations made earlier without a stamp).
    TotalSegmentator writes a file for every class of a task, also for regions
    outside the field of view (empty mask), so missing markers mean the task
    was not run (or crashed).
The stamp prevents endless reruns if a TotalSegmentator version names a
marker region differently.
"""
import os
from pathlib import Path

from .folder_detection import ct_candidates_for_segmentation, check_ct_image
from .Segmentation_toolbox.image_alignment import align_image_orientation

TS_FOLDER = "CT_segmentation"

# task -> files that every complete run of this task writes
TOTALSEG_TASK_MARKERS = {
    "total": ["spleen", "liver", "aorta", "inferior_vena_cava", "vertebrae_L2"],
    "body": ["body"],
    "lung_nodules": ["lung_nodules"],
    "tissue_types": ["subcutaneous_fat", "torso_fat", "skeletal_muscle"],
}


def totalseg_folder(patient_path):
    """Existing segmentation folder (any capitalisation) or the default path."""
    patient_path = Path(patient_path)
    existing = next((d for d in patient_path.iterdir()
                     if d.is_dir() and d.name.lower() == TS_FOLDER.lower()), None)
    return existing or patient_path / TS_FOLDER


def totalseg_task_done(patient_path, task):
    """True/False, or None if the task has no known markers (cannot be checked)."""
    markers = TOTALSEG_TASK_MARKERS.get(task)
    folder = totalseg_folder(patient_path)
    if not folder.is_dir():
        return False
    if (folder / _stamp_name(task)).exists():
        return True
    if markers is None:
        return None
    present = {f.name.lower() for f in folder.iterdir()}
    return all(f"{m.lower()}.nii.gz" in present for m in markers)


def _stamp_name(task):
    return f".totalseg_{task}.done"


def _check_task_output(output_dir, task):
    """Raise if all marker masks of a task are empty (segmentation of a garbage CT)."""
    import SimpleITK as sitk
    files = [Path(output_dir) / f"{m}.nii.gz" for m in TOTALSEG_TASK_MARKERS.get(task, [])]
    files = [f for f in files if f.exists()]
    if files and not any(sitk.GetArrayFromImage(sitk.ReadImage(str(f))).any() for f in files):
        raise RuntimeError(f"TotalSegmentator '{task}' masks are all empty")


def _aligned_ct_path(ct_source):
    """Output path of the orientation-corrected CT: AC_CT.nii.gz -> AC_CT_corrected.nii.gz"""
    p = Path(ct_source["path"])
    stem = p.name
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return p.parent / f"{stem}_corrected.nii.gz"


def totalseg_segment_patient(patient_path, tasks, fast=False, ct_name_hints=None):
    """
    Run the given TotalSegmentator tasks for one patient. The CT is
    orientation-corrected first (written as <name>_corrected.nii.gz if needed).
    Raises on failure.
    """
    from totalsegmentator.python_api import totalsegmentator
    try:
        import dicom2nifti.settings as settings
        settings.disable_validate_slice_increment()
    except ImportError:
        pass

    patient_path = Path(patient_path)
    candidates = ct_candidates_for_segmentation(patient_path, ct_name_hints)
    if not candidates:
        raise FileNotFoundError("no CT found")

    output_dir = totalseg_folder(patient_path)
    os.makedirs(output_dir, exist_ok=True)

    # try the CT candidates in order (e.g. faulty AC_CT DICOMs -> CT NIfTI)
    errors = []
    for ct in candidates:
        rel = os.path.relpath(ct["path"], patient_path)
        try:
            ok, reason = check_ct_image(ct)
            if not ok:
                raise ValueError(f"CT not usable: {reason}")
            if not Path(ct["path"]).name.lower().endswith("corrected.nii.gz"):
                aligned = _aligned_ct_path(ct)
                align_image_orientation(str(ct["path"]), str(aligned))
                if aligned.exists():
                    ct = {"path": aligned, "kind": "nifti", "series_id": None}
            for task in tasks:
                print(f"Running TotalSegmentator task '{task}' on {rel} ...")
                totalsegmentator(ct["path"], output_dir, fast=fast, task=task)
                _check_task_output(output_dir, task)
                (Path(output_dir) / _stamp_name(task)).write_text(f"{ct['path']}\n")
            break
        except Exception as e:
            errors.append(f"{rel} ({ct['kind']}): {e}")
            print(f"❌ TotalSegmentator failed with CT {rel}: {e}")
    else:
        raise RuntimeError("TotalSegmentator failed with every CT: " + " | ".join(errors))
    if errors:
        from .issue_log import log_issue
        log_issue(patient_path.name, "CT",
                  f"TotalSegmentator failed with {'; '.join(errors)} -> segmented with {rel}")
    print(f"✅ {patient_path.name}: TotalSegmentator {tasks}")