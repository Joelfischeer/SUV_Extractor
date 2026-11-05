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
from .segmentation_sources import discover_moose_models, discover_cads_models, get_segmentation_source
from .issue_log import log_issue
from .folder_detection import list_patient_dirs


def _required(backend, moose_models, totalseg_tasks, cads_tasks):
    return list({"moose": moose_models, "cads": cads_tasks or []}.get(backend, totalseg_tasks))


def missing_segmentations(patient_path, backend, moose_models, totalseg_tasks, cads_tasks=None):
    """Models/tasks of `backend` that are missing for this patient."""
    if backend == "moose":
        present = discover_moose_models(patient_path, verbose=False)
        return [m for m in moose_models if m.lower() not in present]
    if backend == "cads":
        present = discover_cads_models(patient_path, verbose=False)
        return [t for t in (cads_tasks or []) if f"cads_{t}" not in present]
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
                         ct_name_hints=None, moose_accelerator=None, totalseg_fast=False,
                         cads_tasks=None, cads_license="research", cads_use_cpu=False,
                         cads_command=None, cads_one_run=True):
    """
    Check (and, if enabled, complete) the segmentations of every patient.

    organ_configs : dict backend -> (organs, combination_logic), optional,
        to report requested organs no segmentation provides.

    Returns a DataFrame: patient, backend, missing_before, action,
    missing_after, organs_not_available.
    """
    data_dir = Path(data_dir)
    patients = list_patient_dirs(data_dir)
    rows = []
    # a tool that fails for several patients in a row (installation, weights,
    # GPU) is not tried again for the rest of the run
    max_failures_in_row = 3
    failures_in_row = {b: 0 for b in backends}
    disabled = {}
    # TotalSegmentator tasks that need a licence the user does not have
    ts_blocked = []
    if "totalsegmentator" in backends and (segment_missing or force):
        from .TotalSegmentator import check_totalseg_license
        ts_blocked, message = check_totalseg_license(list(totalseg_tasks))
        if ts_blocked:
            print("❌ " + message)
            log_issue("ALL", "segmentation", message)

    for i, patient_path in enumerate(patients, 1):
        for backend in backends:
            missing = missing_segmentations(patient_path, backend, moose_models, totalseg_tasks, cads_tasks)
            to_run = _required(backend, moose_models, totalseg_tasks, cads_tasks) if force else missing
            if backend == "totalsegmentator" and ts_blocked:
                to_run = [t for t in to_run if t not in ts_blocked]
            row = {"patient": patient_path.name, "backend": backend,
                   "missing_before": ",".join(map(str, missing)), "action": "complete"}

            required = _required(backend, moose_models, totalseg_tasks, cads_tasks)
            if missing and len(missing) == len(required) and not force and backend not in disabled:
                log_issue(patient_path.name, "segmentation",
                          f"no {backend} segmentation at all"
                          + (f" ({backend} disabled for this run)" if backend in disabled else
                             " -> segmenting now" if segment_missing else " (SEGMENT_MISSING is off)"))
            if backend == "cads" and to_run and (segment_missing or force) and "cads_checked" not in disabled:
                # check the CADS environment once, before the first CADS run
                from .CADS_Segmentator import check_cads_setup
                ok, message = check_cads_setup(cads_command, cads_use_cpu)
                disabled["cads_checked"] = True
                print(("✅ " if ok else "❌ ") + message)
                if not ok:
                    disabled["cads"] = f"setup check failed: {message}"
                    log_issue("ALL", "segmentation", f"CADS not run in this run - {message}")
            if to_run and (segment_missing or force) and backend in disabled:
                row["action"] = f"skipped ({backend} disabled for this run)"
            elif to_run and (segment_missing or force):
                print(f"\n[{i}/{len(patients)}] {patient_path.name}: running {backend} {to_run}")
                try:
                    if backend == "moose":
                        moose_segment_patient(patient_path, to_run, ct_name_hints=ct_name_hints,
                                              accelerator=moose_accelerator, skip_existing=False)
                    elif backend == "cads":
                        from .CADS_Segmentator import cads_segment_patient
                        cads_segment_patient(patient_path, to_run, ct_name_hints=ct_name_hints,
                                             license=cads_license, use_cpu=cads_use_cpu,
                                             command=cads_command, one_run=cads_one_run)
                    else:
                        totalseg_segment_patient(patient_path, to_run, fast=totalseg_fast,
                                                 ct_name_hints=ct_name_hints)
                    row["action"] = f"ran {','.join(map(str, to_run))}"
                    failures_in_row[backend] = 0
                except Exception as e:
                    log_issue(patient_path.name, "segmentation", f"{backend} {to_run} failed: {e}")
                    row["action"] = f"FAILED: {e}"
                    failures_in_row[backend] += 1
                    if failures_in_row[backend] >= max_failures_in_row:
                        disabled[backend] = f"failed for {max_failures_in_row} patients in a row"
                        log_issue("ALL", "segmentation",
                                  f"{backend} failed for {max_failures_in_row} patients in a row - not tried "
                                  f"again in this run. Probably an installation problem; last error: {e}")
                missing = missing_segmentations(patient_path, backend, moose_models, totalseg_tasks, cads_tasks)
            elif missing and backend == "totalsegmentator" and set(missing) <= set(ts_blocked):
                row["action"] = f"missing (no TotalSegmentator licence for {','.join(missing)})"
            elif missing:
                row["action"] = "missing (SEGMENT_MISSING is off)"

            row["missing_after"] = ",".join(map(str, missing))
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
        # identical cases grouped (e.g. a tool disabled for the whole run)
        for (missing, action), cases in incomplete.groupby(["missing_after", "action"], sort=False):
            names = list(cases["patient"])
            if len(names) <= 3:
                for name in names:
                    print(f"   {name}: missing {missing}  ({action})")
            else:
                print(f"   {len(names)} patients ({', '.join(names[:3])}, ...): missing {missing}  ({action})")
        if "organs_not_available" in grp:
            counts = (grp["organs_not_available"].str.split(",").explode()
                      .loc[lambda s: s != ""].value_counts())
            for organ, n in counts.items():
                print(f"   ⚠️ {organ}: not provided for {n}/{len(grp)} patients")
    print("===========================================")