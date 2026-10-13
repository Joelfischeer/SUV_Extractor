"""
Central settings for main.py, compare_segmentations.py and inspect_cohort.py.

To process another cohort, change DATA_DIR here, or pass it on the command line
without editing anything:

    python main.py --data-dir ../../Full_Body_Feature_Extraction/data/Images/Lung_cancer_leipzig
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
DATA_DIR = Path("../../Full_Body_Feature_Extraction/data/Images/Lung_cancer_florenz")
OUTPUT_DIR = Path("../Results")

# ======================= Segmentation =======================
SEGMENTATION_BACKEND = "moose"          # "totalsegmentator" or "moose"

# TotalSegmentator runs (output: <patient>/CT_segmentation)
SEGMENTATE_ORGANS = False
SEGMENTATE_BODY = False
SEGMENTATE_LUNG = False

# MOOSE runs (output: <patient>/moosez-<timestamp>); models already present are skipped
SEGMENTATE_MOOSE = False
MOOSE_MODELS = [
    "clin_ct_organs",             # solid organs, lungs, thyroid, brain, ...
    "clin_ct_cardiac",            # heart chambers, aorta, inferior_vena_cava, ...
    "clin_ct_digestive",          # colon, duodenum, small bowel, esophagus
    "clin_ct_muscles",            # gluteus, iliopsoas, autochthon
    "clin_ct_vertebrae",          # vertebra_L2 (reference level)
    "clin_ct_body_composition",   # skeletal_muscle, subcutaneous_fat, visceral_fat
]
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

# ======================= PET / CT detection =======================
# Only tie-breakers: PET must be an attenuation-corrected PT DICOM series in
# BQML/CNTS, CT a NIfTI/DICOM with 'CT' in its name or Modality CT. If a
# patient has several valid series, the first hint found in the folder name or
# SeriesDescription wins (then the most slices). Hints that do not occur in a
# cohort are ignored, so names of all cohorts can stay in these lists.
PET_NAME_HINTS = ["PET_WB_CORRECTED", "LAST-5-MIN", "5-MIN", "FDG"]
CT_NAME_HINTS = ["AC_CT"]

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


def parse_args(description, backend=False):
    """
    Command-line overrides: --data-dir, --output-dir (and --backend).
    Returns (data_dir, output_dir, backend).
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help=f"cohort folder with one sub-folder per patient (default: {DATA_DIR})")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help=f"results folder (default: {OUTPUT_DIR})")
    if backend:
        parser.add_argument("--backend", choices=["totalsegmentator", "moose"],
                            default=SEGMENTATION_BACKEND,
                            help=f"segmentation to extract from (default: {SEGMENTATION_BACKEND})")
    args = parser.parse_args()

    data_dir = args.data_dir
    if not data_dir.is_dir():
        parser.error(f"cohort folder not found: {data_dir.resolve()}")
    if not any(p.is_dir() for p in data_dir.iterdir()):
        parser.error(f"no patient folders in {data_dir.resolve()}")
    return data_dir, args.output_dir, getattr(args, "backend", SEGMENTATION_BACKEND)