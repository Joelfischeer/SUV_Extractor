
from totalsegmentator.python_api import totalsegmentator
from pathlib import Path
import SimpleITK as sitk
from pathlib import Path
import os
import torch

def TotalSegmentator_dicom_CT(data_root, 
                              segmentate_organs_ct, 
                              segmentate_body_ct, 
                              segmentate_lung_ct,
                              fast):
    """
    Runs TotalSegmentator on all patients in data_root.
    Writes a txt file listing failed patient folders.
    """

    data_root = Path(data_root)
    patient_folders = [p for p in data_root.iterdir() if p.is_dir()]
    failed_patients = []

    for patient_folder in patient_folders:
        print(f"Processing {patient_folder}...")
    
        try:
            corrected_files = list(patient_folder.rglob("*corrected.nii.gz"))

            if corrected_files:
                input_file = corrected_files[0]
                print(f"Using corrected NIfTI: {input_file}")

            else:
                ct_dirs = [
                    d for d in patient_folder.iterdir()
                    if d.is_dir() and "CT" in d.name.upper()
                ]
                if not ct_dirs:
                    print(f"⚠️ No CT folder found for {patient_folder.name}")
                    continue

                input_file = ct_dirs[0]
                print(f"Using CT folder: {input_file}")

            output_dir = patient_folder / "CT_segmentation"
            os.makedirs(output_dir, exist_ok=True)

            # Organ segmentation
            if segmentate_organs_ct:
                totalsegmentator(
                    input_file,
                    output_dir,
                    fast=fast,
                )

            # Lung segmentation
            if segmentate_lung_ct:
                totalsegmentator(
                    input_file,
                    output_dir,
                    fast=fast,
                    task="lung_nodules"
                )

            # Body segmentation
            if segmentate_body_ct:
                totalsegmentator(
                    input_file,
                    output_dir,
                    fast=fast,
                    task="body"
                )

            print(f"✅ Finished {patient_folder.name}")

        except Exception as e:
            print(f"❌ Failed for {patient_folder.name}")
            print(str(e))
            failed_patients.append(
                f"{patient_folder.name} -> {str(e)}"
            )

    # Write failure log
    if failed_patients:
        failure_file = data_root / "totalsegmentator_failed_folders.txt"
        with open(failure_file, "w") as f:
            f.write("TotalSegmentator failed for the following folders:\n\n")
            for item in failed_patients:
                f.write(item + "\n")

        print(f"\nFailure log saved to: {failure_file}")

    print("All patients processed.")

from .Segmentation_toolbox.image_alignment import align_image_orientation


from pathlib import Path
import os

def align_and_segment_images(
    data_dir,
    segmentate_organs_ct=False,
    segmentate_body_ct=False,
    segmentate_lung_ct=False,
    fast=True
):
    """
    Align CT images and optionally run TotalSegmentator.
    If TotalSegmentator fails for a patient, the patient folder
    name will be written to a txt file.
    """

    data_dir = Path(data_dir)
    patients = os.listdir(data_dir)

    for patient in patients:

        patient_path = data_dir / patient
        if not patient_path.is_dir():
            continue
        if "segmentation" in patient.lower():
            continue

        image_files = list(os.listdir(patient_path))

        for image_type in image_files:
            file_path = patient_path / image_type
            fp = str(file_path).lower()

            if (
                'segmentation' not in fp
                and 'corrected' not in fp
                and 'ct' in fp
            ):
                output_path = file_path.parent / f"{file_path.name}_corrected.nii.gz"
                align_image_orientation(file_path, output_path)

    failed_patients = []

    if segmentate_organs_ct or segmentate_body_ct or segmentate_lung_ct:
        print("Segmentating: CT")

        import dicom2nifti.settings as settings
        settings.disable_validate_slice_increment()

        try:
            TotalSegmentator_dicom_CT(
                data_dir,
                segmentate_organs_ct,
                segmentate_body_ct,
                segmentate_lung_ct,
                fast=fast
            )
        except Exception as e:
            print("❌ TotalSegmentator failed.")
            print(str(e))

            # Identify which patient caused failure (best effort)
            # Loop manually per patient to isolate failures
            for patient_folder in [p for p in data_dir.iterdir() if p.is_dir()]:
                try:
                    TotalSegmentator_dicom_CT(
                        patient_folder,
                        segmentate_organs_ct,
                        segmentate_body_ct,
                        segmentate_lung_ct,
                        fast=fast
                    )
                except Exception as patient_error:
                    print(f"❌ Failed for {patient_folder.name}")
                    failed_patients.append(
                        f"{patient_folder.name} -> {str(patient_error)}"
                    )

    # ===========================
    # Write failure log if needed
    # ===========================
    if failed_patients:
        failure_file = data_dir / "totalsegmentator_failed_folders.txt"
        with open(failure_file, "w") as f:
            f.write("TotalSegmentator failed for the following folders:\n\n")
            for item in failed_patients:
                f.write(item + "\n")

        print(f"\nFailure log saved to: {failure_file}")

    print("Alignment + Segmentation completed.")

