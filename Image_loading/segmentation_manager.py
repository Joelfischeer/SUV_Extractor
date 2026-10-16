"""
Make sure every patient has the segmentations the extraction needs.

Required per tool (config.py):
    MOOSE            MOOSE_MODELS    sub-models, e.g. clin_ct_organs, clin_ct_cardiac
    TotalSegmentator TOTALSEG_TASKS  tasks, e.g. total, body

Completeness is checked per model/task, not per organ: MOOSE lists every
label of a model (and TotalSegmentator writes a file per class) even when the
organ lies outside the field of view. An organ is therefore only missing when
its model/task was never run - rerunning for an organ outside the scan range
would not help and would repeat on every run.

Modes
    segment_missing=True   run only the missing models/tasks per patient
    force=True             rerun all required models/tasks for every patient
    both False             only report what is missing

Additionally reported per patient: requested organs that none of the present
segmentations provides (e.g. superior_vena_cava does not exist in MOOSE) -
these need another model/task or must be removed from the organ list.
"""
from pathlib import Path

import pandas as pd

from .MOOSE_Segmentator import moose_segment_patient
from .TotalSegmentator import totalseg_segment_patient, totalseg_task_done
from .segmentation_sources import discover_moose_models, get_segmentation_source
from .issue_log import log_issue


def missing_segmentations(patient_path, backend, moose_models, totalseg_tasks):
    """Models/tasks of `backend` that are missing for this patient."""
    if backend == "moose":
        present = discover_moose_models(patient_path, verbose=False)
        return [m for m in moose_models if m.lower() not in present]
    missing = []
    for task in totalseg_tasks:
        done = totalseg_task_done(patient_path, task)
        if done is None:
            print(f"⚠️ TotalSegmentator task '{task}': completeness cannot be checked "
                  f"(add its marker files to TOTALSEG_TASK_MARKERS) - assumed done")
        elif not done:
            missing.append(task)
    return missing


def unavailable_organs(patient_path, backend, organs, combination_logic):
    """Requested organs that the present segmentation does not provide (exact names)."""
    src = get_segmentation_source(patient_path, backend, None, verbose=False)
    names = {n.lower() for n in src.names()} if src else set()
    out = []
    for organ in organs:
        if organ in combination_logic:
            parts = combination_logic[organ]
            if not all(p.lower() in names for p in parts):
                out.append(organ)
        elif organ.lower() not in names:
            out.append(organ)
    return out


def ensure_segmentations(data_dir, backends, moose_models, totalseg_tasks,
                         segment_missing=True, force=False, organ_configs=None,
                         ct_name_hints=None, moose_accelerator=None, totalseg_fast=False):
    """
    Check (and, if enabled, complete) the segmentations of every patient.

    organ_configs : dict backend -> (organs, combination_logic), optional,
        to report requested organs no segmentation provides.

    Returns a DataFrame: patient, backend, missing_before, action,
    missing_after, organs_not_available.
    """
    data_dir = Path(data_dir)
    patients = sorted(p for p in data_dir.iterdir() if p.is_dir())
    rows = []

    for i, patient_path in enumerate(patients, 1):
        for backend in backends:
            missing = missing_segmentations(patient_path, backend, moose_models, totalseg_tasks)
            to_run = list(moose_models if backend == "moose" else totalseg_tasks) if force else missing
            row = {"patient": patient_path.name, "backend": backend,
                   "missing_before": ",".join(missing), "action": "complete"}

            required = list(moose_models if backend == "moose" else totalseg_tasks)
            if missing and len(missing) == len(required) and not force:
                log_issue(patient_path.name, "segmentation",
                          f"no {backend} segmentation at all"
                          + (" -> segmenting now" if segment_missing else " (SEGMENT_MISSING is off)"))
            if to_run and (segment_missing or force):
                print(f"\n[{i}/{len(patients)}] {patient_path.name}: running {backend} {to_run}")
                try:
                    if backend == "moose":
                        moose_segment_patient(patient_path, to_run, ct_name_hints=ct_name_hints,
                                              accelerator=moose_accelerator, skip_existing=False)
                    else:
                        totalseg_segment_patient(patient_path, to_run, fast=totalseg_fast,
                                                 ct_name_hints=ct_name_hints)
                    row["action"] = f"ran {','.join(to_run)}"
                except Exception as e:
                    log_issue(patient_path.name, "segmentation", f"{backend} {to_run} failed: {e}")
                    row["action"] = f"FAILED: {e}"
                missing = missing_segmentations(patient_path, backend, moose_models, totalseg_tasks)
            elif missing:
                row["action"] = "missing (SEGMENT_MISSING is off)"

            row["missing_after"] = ",".join(missing)
            if organ_configs and backend in organ_configs:
                organs, combination = organ_configs[backend]
                row["organs_not_available"] = ",".join(
                    unavailable_organs(patient_path, backend, organs, combination))
            rows.append(row)

    status = pd.DataFrame(rows)
    _print_summary(status, len(patients))
    return status


def _print_summary(status, n_patients):
    print("\n=========== Segmentation status ===========")
    for backend, grp in status.groupby("backend"):
        complete = (grp["missing_after"] == "").sum()
        ran = grp["action"].str.startswith("ran").sum()
        failed = grp["action"].str.startswith("FAILED").sum()
        print(f"{backend}: {complete}/{n_patients} complete, {ran} completed now, {failed} failed")
        incomplete = grp[grp["missing_after"] != ""]
        for _, r in incomplete.iterrows():
            print(f"   {r['patient']}: missing {r['missing_after']}  ({r['action']})")
        if "organs_not_available" in grp:
            counts = (grp["organs_not_available"].str.split(",").explode()
                      .loc[lambda s: s != ""].value_counts())
            for organ, n in counts.items():
                print(f"   ⚠️ {organ}: not provided for {n}/{len(grp)} patients")
    print("===========================================")