"""
Load PET and CT data for every patient of a cohort, segment organs on the CT
(TotalSegmentator or MOOSE), get cropped PET volumes along with patient
metadata, calculate SUV mean, max and min per organ.

All settings are in config.py. The cohort folder can be given on the command
line, so another cohort needs no code change:

    python main.py                                   # DATA_DIR from config.py
    python main.py --data-dir <cohort folder>
    python main.py --data-dir <cohort folder> --backend totalsegmentator

Output: <OUTPUT_DIR>/<cohort>_SUV_extraction/<cohort>_<backend>_SUVs.csv

Cohort folder
-------------
One sub-folder per patient. PET and CT are found automatically
(Image_loading/folder_detection.py):
- PET: attenuation-corrected PT DICOM series in BQML/CNTS. NAC, MIP and other
  PET series are rejected; PET_NAME_HINTS decide between several valid ones.
- CT:  NIfTI with a 'CT' token in its name or a CT DICOM series;
  CT_NAME_HINTS decide between several.
The chosen PET/CT is printed per patient - check it on the first run of a new
cohort (or run inspect_cohort.py first).

Segmentation backends (never mixed)
-----------------------------------
- totalsegmentator: <patient>/CT_segmentation/<region>.nii.gz
- moose:            <patient>/moosez-*/segmentations/ (multi-label files, label
                    names from MOOSE's *organ_indices.json; newest run per
                    MOOSE sub-model)
Organ names must match exactly (or be a combination defined in
Analysis/organ_definitions.py); missing organs give NaN.

NORMALIZE: organ values divided by the mean SUV of the inferior vena cava at
the L2 level (eroded in-plane). reference_SUV_mean / reference_method are
always written to the CSV.
ERODE: EROSION_CONFIG voxels removed around every mask before the SUVs.
"""
import config as cfg
from Analysis.organ_definitions import organ_config


def main():
    data_dir, output_dir, backend = cfg.parse_args(
        "SUV extraction per organ from PET using CT segmentations", backend=True)
    print(f"Cohort: {data_dir.resolve()}\nBackend: {backend}")

    organs_of_interest, combination_logic = organ_config(backend)

    # Results file: <OUTPUT_DIR>/<cohort>_SUV_extraction/<cohort>_<backend>_SUVs.csv
    out_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
    output_path = out_dir / f"{cohort}_{backend}"

    # ======================= Segmentation =======================
    if cfg.SEGMENTATE_ORGANS or cfg.SEGMENTATE_BODY or cfg.SEGMENTATE_LUNG:
        from Image_loading.TotalSegmentator import align_and_segment_images
        align_and_segment_images(
            data_dir=data_dir,
            segmentate_organs_ct=cfg.SEGMENTATE_ORGANS,
            segmentate_body_ct=cfg.SEGMENTATE_BODY,
            segmentate_lung_ct=cfg.SEGMENTATE_LUNG,
            fast=False,
            ct_name_hints=cfg.CT_NAME_HINTS,
        )

    if cfg.SEGMENTATE_MOOSE:
        from Image_loading.MOOSE_Segmentator import moose_segment_images
        moose_segment_images(
            data_dir=data_dir,
            models=cfg.MOOSE_MODELS,
            ct_name_hints=cfg.CT_NAME_HINTS,
            accelerator=cfg.MOOSE_ACCELERATOR,
        )

    # ======================= SUV extraction =======================
    patients = sorted(p.name for p in data_dir.iterdir() if p.is_dir())

    from Image_loading.Image_Loader import PET_Organ_Cropper, erode_organ_masks
    from Analysis.SUV_calculation import compute_suv
    from Results_saving.results_saver import results_saver

    all_results = []
    failed = []

    for patient in patients:
        # 1. Load cropped organs (+ reference, restricted to L2)
        try:
            result = PET_Organ_Cropper(
                data_dir=data_dir,
                patient=patient,
                organs_of_interest=organs_of_interest,
                combination_logic=combination_logic,
                norm_reference=cfg.REFERENCE_ORGAN,
                reference_vertebra=cfg.REFERENCE_VERTEBRA,
                segmentation_backend=backend,
                pet_name_hints=cfg.PET_NAME_HINTS,
                moose_model_priority=cfg.MOOSE_SUBMODEL_PRIORITY,
            )
        except Exception as e:
            print(f"❌ {patient}: {e}")
            failed.append(patient)
            continue

        if result is None:
            failed.append(patient)
            continue

        # 2. Extract organ data and patient metrics
        PET_organs_raw = result["organs"]
        reference_value = result["reference_SUV_mean"]

        # 3. Remove pixels around each organ to remove contamination from bad segmentation
        if cfg.ERODE:
            PET_organs_raw = erode_organ_masks(PET_organs_raw, erosion_config=cfg.EROSION_CONFIG)

        # 4. Normalize to the blood pool (IVC at L2)
        if cfg.NORMALIZE:
            if reference_value is None or reference_value <= 0:
                print(f"⚠️ No valid {cfg.REFERENCE_ORGAN} reference for {patient} - normalized values set to NaN")
                PET_organs_normalized = {}
            else:
                PET_organs_normalized = {organ: arr / reference_value
                                         for organ, arr in PET_organs_raw.items()}
        else:
            PET_organs_normalized = PET_organs_raw

        # 5. Compute SUVs
        patient_result = compute_suv(
            PET_organs_normalized,
            organs_of_interest=organs_of_interest + [cfg.REFERENCE_ORGAN],
            patient_id=patient
        )

        patient_result['height_m'] = result["patient_height_m"]
        patient_result['weight_kg'] = result["patient_weight_kg"]
        patient_result['bmi'] = result["patient_BMI"]
        patient_result['time_of_scan'] = result["patient_time_of_scan"]
        patient_result['uptake_time_min'] = result["suv_conversion"].get("uptake_time_min")
        patient_result['normalized'] = cfg.NORMALIZE
        patient_result['reference_SUV_mean'] = reference_value
        patient_result['reference_method'] = result["reference_method"]
        patient_result['reference_voxels'] = result["reference_voxels"]
        patient_result['segmentation_backend'] = result["segmentation_backend"]
        patient_result['pet_series'] = result["pet_series"]
        all_results.append(patient_result)

    if failed:
        print(f"\n⚠️ {len(failed)} patients without results: {failed}")

    if not all_results:
        print("No results - nothing saved.")
        return
    results_saver(all_results, output_path)


if __name__ == "__main__":
    main()