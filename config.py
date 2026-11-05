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
#   <cohort>_totalsegmentator_SUVs.csv / <cohort>_moose_SUVs.csv / <cohort>_cads_SUVs.csv
EXTRACT_TOTALSEGMENTATOR = False
EXTRACT_MOOSE = True
EXTRACT_CADS = False
# Compare the tools (SUV_mean/max, no erosion vs 1 voxel eroded) -> tool_comparison/
# with one sub-folder per pair (TS_vs_MOOSE, TS_vs_CADS, MOOSE_vs_CADS).
# Needs the segmentations of every compared tool; missing ones are completed if
# SEGMENT_MISSING = True.
COMPARE_TOOLS = True
COMPARISON_TOOLS = ["totalsegmentator", "moose", "cads"]

# ======================= Segmentation =======================

# ---- Required segmentations per tool ----
# Before the extraction every patient is checked against these lists.
# TotalSegmentator tasks (output: <patient>/CT_segmentation); known tasks:
# "total", "body", "lung_nodules", "tissue_types" (see TOTALSEG_TASK_MARKERS
# in Image_loading/Segmentator.py to add others)
# "tissue_types" (subcutaneous_fat, torso_fat -> used as visceral_fat,
# skeletal_muscle) needs a TotalSegmentator licence, free for academic use:
#   https://backend.totalsegmentator.com/license-academic/
# set once with:  uv run totalseg_set_license -l <licence number>
# Without a licence it is skipped (logged) and 'total' still runs.
TOTALSEG_TASKS = ["total", "tissue_types"]
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

# CADS tasks (output: <patient>/cads_segmentation); https://github.com/murong-xu/CADS
#   551 abdominal organs, lungs, aorta, IVC   552 vertebrae (L2)   553 heart chambers,
#   brain, bowel   554 bones, gluteal muscles   555 ribs   556 heart, radiotherapy OARs
#   557 head tissues   558 head & neck incl. thyroid   559 subcutaneous tissue, muscle
CADS_TASKS = [551, 552, 553, 554, 556, 558, 559]
# Model weights: "reference" (CADS custom licence), "research" (CC BY-NC-SA 4.0,
# recommended by the authors for academic use), "open" (CC BY-SA 4.0)
CADS_LICENSE = "research"
CADS_USE_CPU = False
# CADS pins nnunetv2==2.5.2 and is best installed in its own environment. Then
# set the command that runs its predict script there, e.g.
#   ["conda", "run", "-n", "CADS_env", "python", "/path/to/CADS/cads/scripts/predict_images.py"]
# None: use the 'cads' package of this environment.
CADS_COMMAND = None
# True: all CADS tasks of a patient in one run (one preprocessing of the CT,
# ~40 % faster); falls back to one call per task if that fails.
CADS_ONE_RUN_PER_PATIENT = True

MOOSE_ACCELERATOR = None              # "cuda", "cpu", "mps" or None (auto)
# Only within MOOSE: if a region is produced by several MOOSE sub-models
# (e.g. aorta in clin_ct_organs and clin_ct_cardiac), the one listed first is
# used. TotalSegmentator and MOOSE are never mixed.
MOOSE_SUBMODEL_PRIORITY = MOOSE_MODELS

# ======================= SUV extraction =======================
NORMALIZE = False
ERODE = False

# Blood-pool reference
REFERENCE_ORGAN = "inferior_vena_cava"
REFERENCE_VERTEBRA = "vertebrae_L2"

# Erosion in voxels per organ (DEFAULT for all others)
EROSION_CONFIG = {'DEFAULT': 0}

# Outlier protection: if any organ SUV_mean exceeds this with the DICOM PET
# (or the PET DICOMs cannot be read), the patient is written to
# <cohort>_patient_issues.txt and the PET is loaded from the patient's PET
# NIfTI instead (file with 'wb_pet' / 'pet' in the name). Its unit (scanner
# units or SUV) is decided by plausibility. All tables and the comparison use
# the same PET source per patient (column pet_source).
SUV_MEAN_OUTLIER_THRESHOLD = 20.0
PET_NIFTI_HINTS = ["WB_PET", "PET"]

# Patient filter: exclude patients whose organ SUV_means deviate from the
# cohort distribution by more than OUTLIER_FILTER_SD (robust) SDs in at least
# OUTLIER_FILTER_MIN_ORGANS organs (e.g. values consistently 10x too high).
# Excluded patients are removed from the SUV tables and the comparison, and
# listed with the reason in <cohort>_excluded_patients.csv and the issue log.
OUTLIER_FILTER = True
OUTLIER_FILTER_SD = 2.0
OUTLIER_FILTER_MIN_ORGANS = 5
OUTLIER_FILTER_METHOD = "robust"      # "robust" (median/MAD) or "sd" (mean/SD)

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
    resolved = Path(data_dir).resolve()
    cohort = resolved.name
    if cohort.lower() in {"train", "test", "val", "validation", "rawdata", "data"}:
        cohort = f"{resolved.parent.name}_{cohort}"     # e.g. Multimodal_HC_train
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
        parser.add_argument("--backend", nargs="+", default=None,
                            choices=["totalsegmentator", "moose", "cads", "both", "all"],
                            help="tool(s) to extract, e.g. '--backend moose cads'; 'all' = all three, "
                                 "'both' = totalsegmentator + moose; default: EXTRACT_* in config.py")
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
        all_tools = ["totalsegmentator", "moose", "cads"]
        if args.backend is None:
            backends = [b for b, on in zip(all_tools, (EXTRACT_TOTALSEGMENTATOR, EXTRACT_MOOSE,
                                                       EXTRACT_CADS)) if on]
        else:
            chosen = set(args.backend)
            if "all" in chosen:
                chosen |= set(all_tools)
            if "both" in chosen:
                chosen |= {"totalsegmentator", "moose"}
            backends = [b for b in all_tools if b in chosen]
        compare = COMPARE_TOOLS if args.compare is None else args.compare
        if not backends and not compare:
            parser.error("nothing to do: switch on EXTRACT_TOTALSEGMENTATOR, EXTRACT_MOOSE, "
                         "EXTRACT_CADS or COMPARE_TOOLS in config.py")
        options = {"backends": backends, "compare": compare}
    return data_dir, args.output_dir, options