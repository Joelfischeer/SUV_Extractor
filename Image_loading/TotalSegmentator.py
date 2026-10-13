import os
from pathlib import Path

from .folder_detection import find_ct_source
from .Segmentation_toolbox.image_alignment import align_image_orientation


def _aligned_ct_path(ct_source):
    """Output path of the orientation-corrected CT: AC_CT.nii.gz -> AC_CT_corrected.nii.gz"""
    p = Path(ct_source["path"])
    stem = p.name
    for ext in (".nii.gz", ".nii"):
        if stem.lower().endswith(ext):
            stem = stem[: -len(ext)]
    return p.parent / f"{stem}_corrected.nii.gz"


def TotalSegmentator_dicom_CT(data_root,
                              segmentate_organs_ct,
                              segmentate_body_ct,
                              segmentate_lung_ct,
                              fast,
                              ct_name_hints=None):
    """
    Runs TotalSegmentator on all patients in data_root.
    The CT is located with find_ct_source (aligned '*_corrected.nii.gz' first).
    Writes a txt file listing failed patient folders.
    """
    from totalsegmentator.python_api import totalsegmentator

    data_root = Path(data_root)
    patient_folders = sorted(p for p in data_root.iterdir() if p.is_dir())
    failed_patients = []

    for patient_folder in patient_folders:
        print(f"Processing {patient_folder}...")

        try:
            ct = find_ct_source(patient_folder, name_hints=ct_name_hints)
            if ct is None:
                print(f"⚠️ No CT found for {patient_folder.name}")
                continue
            input_file = ct["path"]

            output_dir = patient_folder / "CT_segmentation"
            os.makedirs(output_dir, exist_ok=True)

            # Organ segmentation
            if segmentate_organs_ct:
                totalsegmentator(input_file, output_dir, fast=fast)

            # Lung segmentation
            if segmentate_lung_ct:
                totalsegmentator(input_file, output_dir, fast=fast, task="lung_nodules")

            # Body segmentation
            if segmentate_body_ct:
                totalsegmentator(input_file, output_dir, fast=fast, task="body")

            print(f"✅ Finished {patient_folder.name}")

        except Exception as e:
            print(f"❌ Failed for {patient_folder.name}")
            print(str(e))
            failed_patients.append(f"{patient_folder.name} -> {str(e)}")

    # Write failure log
    if failed_patients:
        failure_file = data_root / "totalsegmentator_failed_folders.txt"
        with open(failure_file, "w") as f:
            f.write("TotalSegmentator failed for the following folders:\n\n")
            for item in failed_patients:
                f.write(item + "\n")

        print(f"\nFailure log saved to: {failure_file}")

    print("All patients processed.")


def align_and_segment_images(
    data_dir,
    segmentate_organs_ct=False,
    segmentate_body_ct=False,
    segmentate_lung_ct=False,
    fast=True,
    ct_name_hints=None,
):
    """
    Align the CT of every patient (only the CT selected by find_ct_source,
    written as <name>_corrected.nii.gz if the orientation is non-standard)
    and optionally run TotalSegmentator. Failures are logged per patient.
    """
    data_dir = Path(data_dir)

    for patient_path in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        if "segmentation" in patient_path.name.lower():
            continue
        ct = find_ct_source(patient_path, name_hints=ct_name_hints, verbose=False)
        if ct is None:
            print(f"⚠️ No CT found for {patient_path.name} - not aligned")
            continue
        if Path(ct["path"]).name.lower().endswith("corrected.nii.gz"):
            continue                                    # already aligned earlier
        try:
            align_image_orientation(str(ct["path"]), str(_aligned_ct_path(ct)))
        except Exception as e:
            print(f"⚠️ Alignment failed for {patient_path.name}: {e}")

    if segmentate_organs_ct or segmentate_body_ct or segmentate_lung_ct:
        print("Segmentating: CT")

        import dicom2nifti.settings as settings
        settings.disable_validate_slice_increment()

        # failures are caught and logged per patient inside
        TotalSegmentator_dicom_CT(
            data_dir,
            segmentate_organs_ct,
            segmentate_body_ct,
            segmentate_lung_ct,
            fast=fast,
            ct_name_hints=ct_name_hints,
        )

    print("Alignment + Segmentation completed.")