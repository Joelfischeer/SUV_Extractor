"""
Patient issue log: <OUTPUT_DIR>/<cohort>_SUV_extraction/<cohort>_patient_issues.txt

Every problem that changes how a patient is processed (PET outliers, PET
loaded from NIfTI instead of DICOM, CT DICOM failing for the segmentation,
...) is written here, one line per issue:

    2026-10-14 10:22 | FI003DT | PET | SUV_mean > 20 with DICOM PET: kidneys 35.2 -> ...

The same message is written only once per run (main.py runs several
extractions and the comparison for the same patient).
"""
from datetime import datetime
from pathlib import Path

_LOG_PATH = None
_SEEN = set()


def set_issue_log(path, reset=False):
    """Write issues to `path`. reset=True starts a new file (beginning of a run)."""
    global _LOG_PATH
    _LOG_PATH = Path(path)
    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if reset or not _LOG_PATH.exists():
        _SEEN.clear()
        _LOG_PATH.write_text("Patient issues (date | patient | area | message)\n\n")


def log_issue(patient, area, message):
    """Print and append one issue (once per run)."""
    key = (patient, area, message)
    print(f"❗ {patient} [{area}]: {message}")
    if key in _SEEN:
        return
    _SEEN.add(key)
    if _LOG_PATH is not None:
        with open(_LOG_PATH, "a") as f:
            f.write(f"{datetime.now():%Y-%m-%d %H:%M} | {patient} | {area} | {message}\n")