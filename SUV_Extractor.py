"""
Load PET and CT data for every patient of a cohort, segment organs on the CT
(TotalSegmentator or MOOSE), get cropped PET volumes along with patient
metadata, calculate SUV mean, max and min per organ.

All settings are in config.py. The cohort folder can be given on the command
line, so another cohort needs no code change:

    python main.py                                   # everything switched on in config.py
    python main.py --data-dir <cohort folder>
    python main.py --backend moose --no-compare      # override the switches once

What runs is set in config.py:
    EXTRACT_TOTALSEGMENTATOR  -> <cohort>_totalsegmentator_SUVs.csv
    EXTRACT_MOOSE             -> <cohort>_moose_SUVs.csv
    COMPARE_TOOLS             -> TS_vs_MOOSE/ (agreement statistics and plots)
all in <OUTPUT_DIR>/<cohort>_SUV_extraction/. The tools are never mixed:
each table comes from one tool only.

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

Segmentations
-------------
Before the extraction every patient is checked against the required MOOSE
models / TotalSegmentator tasks (config.py). With SEGMENT_MISSING = True only
the missing ones are run; FORCE_RESEGMENTATION = True reruns all of them.
Status per patient: <cohort>_<backend>_segmentation_status.csv

Segmentation backends (never mixed)
-----------------------------------
- totalsegmentator: <patient>/CT_segmentation/<region>.nii.gz
- moose:            <patient>/moosez-*/segmentations/ (multi-label files, label
                    names from MOOSE's *organ_indices.json; newest run per
                    MOOSE sub-model)
Organ names must match exactly (or be a combination defined in
Analysis/organ_definitions.py); missing organs give NaN.

Outliers: if an organ SUV_mean exceeds SUV_MEAN_OUTLIER_THRESHOLD (20) with
the DICOM PET, the patient is logged in <cohort>_patient_issues.txt and the PET
is loaded from the patient's PET NIfTI instead (columns pet_source, pet_issue).
All tables use the same PET source per patient.

NORMALIZE: organ values divided by the mean SUV of the inferior vena cava at
the L2 level (eroded in-plane). reference_SUV_mean / reference_method are
always written to the CSV.
ERODE: EROSION_CONFIG voxels removed around every mask before the SUVs.
"""
import config as cfg
from Analysis.organ_definitions import organ_config


def extract_patient_row(data_dir, patient, backend, organs_of_interest, combination_logic):
    """One result row (dict) for one patient and one tool, or None."""
    from Image_loading.Image_Loader import PET_Organ_Cropper, erode_organ_masks
    from Analysis.SUV_calculation import compute_suv

    # 1. Load cropped organs (+ reference, restricted to L2)
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
        suv_outlier_threshold=cfg.SUV_MEAN_OUTLIER_THRESHOLD,
        pet_nifti_hints=cfg.PET_NIFTI_HINTS,
    )
    if result is None:
        return None

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
    row = compute_suv(
        PET_organs_normalized,
        organs_of_interest=organs_of_interest + [cfg.REFERENCE_ORGAN],
        patient_id=patient
    )
    row['height_m'] = result["patient_height_m"]
    row['weight_kg'] = result["patient_weight_kg"]
    row['bmi'] = result["patient_BMI"]
    row['time_of_scan'] = result["patient_time_of_scan"]
    row['uptake_time_min'] = result["suv_conversion"].get("uptake_time_min")
    row['normalized'] = cfg.NORMALIZE
    row['reference_SUV_mean'] = reference_value
    row['reference_method'] = result["reference_method"]
    row['reference_voxels'] = result["reference_voxels"]
    row['segmentation_backend'] = result["segmentation_backend"]
    row['pet_series'] = result["pet_series"]
    row['pet_source'] = result["pet_source"]
    row['pet_issue'] = result["pet_issue"]
    return row


def run_extraction(data_dir, output_dir, backend):
    """
    Result rows of one cohort for ONE segmentation tool (never mixed).
    Returns (rows, output_path); saved by main() after the consistency check.
    """
    print(f"\n#################### SUV extraction: {backend} ####################")
    print(f"Cohort: {data_dir.resolve()}")

    organs_of_interest, combination_logic = organ_config(backend)

    # Results: <OUTPUT_DIR>/<cohort>_SUV_extraction/<cohort>_<backend>_SUVs.csv
    out_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
    output_path = out_dir / f"{cohort}_{backend}"

    # ======================= Segmentation =======================
    # Check every patient against the required models/tasks (config.py);
    # run only what is missing if SEGMENT_MISSING is on.
    from Image_loading.segmentation_manager import ensure_segmentations
    status = ensure_segmentations(
        data_dir, [backend],
        moose_models=cfg.MOOSE_MODELS,
        totalseg_tasks=cfg.TOTALSEG_TASKS,
        segment_missing=cfg.SEGMENT_MISSING,
        force=cfg.FORCE_RESEGMENTATION,
        organ_configs={backend: (organs_of_interest + [cfg.REFERENCE_ORGAN, cfg.REFERENCE_VERTEBRA],
                                 combination_logic)},
        ct_name_hints=cfg.CT_NAME_HINTS,
        moose_accelerator=cfg.MOOSE_ACCELERATOR,
        totalseg_fast=cfg.TOTALSEG_FAST,
    )
    status.to_csv(out_dir / f"{cohort}_{backend}_segmentation_status.csv", index=False, sep=";")

    # ======================= SUV extraction =======================
    rows, failed = [], []
    for patient in sorted(p.name for p in data_dir.iterdir() if p.is_dir()):
        try:
            row = extract_patient_row(data_dir, patient, backend, organs_of_interest, combination_logic)
        except Exception as e:
            print(f"❌ {patient}: {e}")
            row = None
        if row is None:
            failed.append(patient)
        else:
            rows.append(row)

    if failed:
        print(f"\n⚠️ {backend}: {len(failed)} patients without results: {failed}")
    return rows, output_path


def make_pet_sources_consistent(data_dir, tables):
    """
    If a patient switched to the PET NIfTI during a later extraction, re-extract
    it in the tables that still used the DICOM PET (same PET in every table).
    """
    from Image_loading.Image_Loader import pet_source_decision
    for backend, (rows, _) in tables.items():
        organs_of_interest, combination_logic = organ_config(backend)
        for i, row in enumerate(rows):
            if row.get("pet_source") == "dicom" and pet_source_decision(data_dir / row["Patient"]):
                print(f"\n↻ {row['Patient']}: re-extracting {backend} with the PET NIfTI")
                new = extract_patient_row(data_dir, row["Patient"], backend,
                                          organs_of_interest, combination_logic)
                if new is not None:
                    rows[i] = new


def main():
    from Image_loading.issue_log import set_issue_log
    from Image_loading.Image_Loader import reset_pet_source_decisions
    from Results_saving.results_saver import results_saver

    data_dir, output_dir, options = cfg.parse_args(
        "SUV extraction per organ from PET using CT segmentations", run_options=True)
    print(f"Cohort: {data_dir.resolve()}\n"
          f"Extraction: {', '.join(options['backends']) or 'none'} | "
          f"comparison: {'yes' if options['compare'] else 'no'}")

    out_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
    set_issue_log(out_dir / f"{cohort}_patient_issues.txt", reset=True)
    reset_pet_source_decisions()

    tables = {backend: run_extraction(data_dir, output_dir, backend)
              for backend in options["backends"]}
    make_pet_sources_consistent(data_dir, tables)
    for backend, (rows, output_path) in tables.items():
        if rows:
            results_saver(rows, output_path)
        else:
            print(f"No {backend} results - nothing saved.")

    if options["compare"]:
        from Analysis.compare_segmentations import run_comparison
        run_comparison(data_dir, output_dir)

    print(f"\nPatient issues: {out_dir / f'{cohort}_patient_issues.txt'}")


if __name__ == "__main__":
    main()