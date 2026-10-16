"""
Central settings for main.py, compare_segmentations.py and inspect_cohort.py.

To process another cohort, change DATA_DIR here, or pass it on the command line
without editing anything:

    python main.py --data-dir ../../Full_Body_Feature_Extraction/data/Images/Lung_cancer_leipzig
    python main.py --backend moose --no-compare      (override the switches below)
    python compare_segmentations.py --data-dir <cohort folder>
    python inspect_cohort.py --data-dir <cohort folder>

All results of a cohort go to <OUTPUT_DIR>/<cohort>_SUV_extraction/, e.g.
    ../Results/Lung_cancer_leipzig_SUV_extraction/
        Lung_cancer_leipzig_moose_SUVs.csv
        Lung_cancer_leipzig_totalsegmentator_SUVs.csv
        Lung_cancer_leipzig_inspection.csv
        TS_vs_MOOSE/                       (compare_segmentations.py)

Expected layout of a cohort folder: one sub-folder per patient containing the
PET DICOM series and the CT (NIfTI or DICOM) anywhere inside it, plus
CT_segmentation/ (TotalSegmentator) and/or moosez-*/ (MOOSE). PET and CT are
found from the DICOM headers, so folder names don't matter.
"""
import argparse
from pathlib import Path

# ======================= Cohort =======================
DATA_DIR = Path("../../Full_Body_Feature_Extraction/data/Images/Lung_cancer_leipzig")
OUTPUT_DIR = Path("../Results")

# ======================= What to run (main.py) =======================
# Each tool gives its own result table (never mixed):
#   <cohort>_totalsegmentator_SUVs.csv  /  <cohort>_moose_SUVs.csv
EXTRACT_TOTALSEGMENTATOR = True
EXTRACT_MOOSE = True
# Compare both tools (SUV_mean/max, no erosion vs 1 voxel eroded) -> TS_vs_MOOSE/
# Needs both segmentations; missing ones are completed if SEGMENT_MISSING = True.
COMPARE_TOOLS = True

# ======================= Segmentation =======================

# ---- Required segmentations per tool ----
# Before the extraction every patient is checked against these lists.
# TotalSegmentator tasks (output: <patient>/CT_segmentation); known tasks:
# "total", "body", "lung_nodules", "tissue_types" (see TOTALSEG_TASK_MARKERS
# in Image_loading/Segmentator.py to add others)
TOTALSEG_TASKS = ["total"]
TOTALSEG_FAST = False

# MOOSE sub-models (output: <patient>/moosez-<timestamp>)
MOOSE_MODELS = [
    "clin_ct_organs",             # liver, spleen, kidneys, pancreas, stomach, thyroid, brain, ...
    "clin_ct_cardiac",            # heart chambers, aorta, inferior_vena_cava, ...
    "clin_ct_digestive",          # colon, duodenum, small bowel, esophagus
    "clin_ct_muscles",            # gluteus, iliopsoas, autochthon
    "clin_ct_vertebrae",          # vertebra_L2 (reference level)
    "clin_ct_body_composition",   # skeletal_muscle, subcutaneous_fat, visceral_fat
    # "clin_ct_lungs",            # lung lobes (needed for 'lungs')
    # "clin_ct_ribs", "clin_ct_peripheral_bones",
]

# ---- What to do with missing segmentations ----
# True:  run only the models/tasks a patient is missing, then extract.
#        Existing segmentations are never redone.
# False: only report what is missing (organs from missing models are NaN).
SEGMENT_MISSING = True
# True: rerun ALL required models/tasks for every patient (e.g. after a model update)
FORCE_RESEGMENTATION = False

MOOSE_ACCELERATOR = None              # "cuda", "cpu", "mps" or None (auto)
# Only within MOOSE: if a region is produced by several MOOSE sub-models
# (e.g. aorta in clin_ct_organs and clin_ct_cardiac), the one listed first is
# used. TotalSegmentator and MOOSE are never mixed.
MOOSE_SUBMODEL_PRIORITY = MOOSE_MODELS

# ======================= SUV extraction =======================
NORMALIZE = False
ERODE = True

# Blood-pool reference
REFERENCE_ORGAN = "inferior_vena_cava"
REFERENCE_VERTEBRA = "vertebrae_L2"

# Erosion in voxels per organ (DEFAULT for all others)
EROSION_CONFIG = {'DEFAULT': 1}

# Outlier protection: if any organ SUV_mean exceeds this with the DICOM PET
# (or the PET DICOMs cannot be read), the patient is written to
# <cohort>_patient_issues.txt and the PET is loaded from the patient's PET
# NIfTI instead (file with 'wb_pet' / 'pet' in the name). Its unit (scanner
# units or SUV) is decided by plausibility. All tables and the comparison use
# the same PET source per patient (column pet_source).
SUV_MEAN_OUTLIER_THRESHOLD = 20.0
PET_NIFTI_HINTS = ["WB_PET", "PET"]

# ======================= PET / CT detection =======================
# Only tie-breakers: PET must be an attenuation-corrected PT DICOM series in
# BQML/CNTS, CT a NIfTI/DICOM with 'CT' in its name or Modality CT. If a
# patient has several valid series, the first hint found in the folder name or
# SeriesDescription wins (then the most slices). Hints that do not occur in a
# cohort are ignored, so names of all cohorts can stay in these lists.
PET_NAME_HINTS = ["PET_WB_CORRECTED", "LAST-5-MIN", "5-MIN", "FDG"]
CT_NAME_HINTS = ["AC_CT"]
# If a segmentation fails with the chosen CT (e.g. faulty AC_CT DICOMs), it is
# retried with the next CT candidate - NIfTI files with 'CT' in the name first.

# ======================= Comparison =======================
COMPARISON_EROSION_VOXELS = 1         # compared against no erosion
COMPARE_ALL_SHARED_REGIONS = False    # True: every region present in both tools


def cohort_output_dir(output_dir, data_dir):
    """All results of a cohort: <OUTPUT_DIR>/<cohort folder name>_SUV_extraction/"""
    from pathlib import Path
    cohort = Path(data_dir).resolve().name
    out = Path(output_dir) / f"{cohort}_SUV_extraction"
    out.mkdir(parents=True, exist_ok=True)
    return out, cohort


def parse_args(description, run_options=False):
    """
    Command-line overrides: --data-dir, --output-dir; for main.py also
    --backend {totalsegmentator,moose,both} and --compare / --no-compare.
    Returns (data_dir, output_dir, options); options is None unless
    run_options, else {"backends": [...], "compare": bool}.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"cohort folder with one sub-folder per patient (default: {DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"results folder (default: {OUTPUT_DIR})")
    if run_options:
        parser.add_argument("--backend", choices=["totalsegmentator", "moose", "both"], default=None,
                            help="tool(s) to extract; default: EXTRACT_TOTALSEGMENTATOR / EXTRACT_MOOSE")
        parser.add_argument("--compare", action=argparse.BooleanOptionalAction, default=None,
                            help=f"compare both tools (default: COMPARE_TOOLS = {COMPARE_TOOLS})")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        parser.error(f"cohort folder not found: {data_dir.resolve()}")
    if not any(p.is_dir() for p in data_dir.iterdir()):
        parser.error(f"no patient folders in {data_dir.resolve()}")

    options = None
    if run_options:
        if args.backend is None:
            backends = [b for b, on in (("totalsegmentator", EXTRACT_TOTALSEGMENTATOR),
                                        ("moose", EXTRACT_MOOSE)) if on]
        elif args.backend == "both":
            backends = ["totalsegmentator", "moose"]
        else:
            backends = [args.backend]
        compare = COMPARE_TOOLS if args.compare is None else args.compare
        if not backends and not compare:
            parser.error("nothing to do: switch on EXTRACT_TOTALSEGMENTATOR, EXTRACT_MOOSE "
                         "or COMPARE_TOOLS in config.py")
        options = {"backends": backends, "compare": compare}
    return data_dir, args.output_dir, options