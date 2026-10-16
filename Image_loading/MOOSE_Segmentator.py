"""
Run MOOSE (https://github.com/ENHANCE-PET/MOOSE, pip install moosez) on the
CT of every patient.

Output per patient, same layout as the MOOSE command line tool:

    <patient>/moosez-YYYY-MM-DD-HH-MM-SS/
        MOOSE_information.json
        segmentations/<prefix>segmentation_<ct>.nii.gz   (multi-label, one per model)
        segmentations/<prefix>organ_indices.json          (label value -> region name)
        stats/

The command line tool expects files called CT_*.nii.gz directly in the
patient folder; here the CT is located with find_ct_source (e.g.
Nifti/AC_CT.nii.gz or a CT DICOM series) and passed to the moosez python API,
so nothing has to be renamed or copied.

Models that already exist in an earlier moosez-* folder of the patient are not
run again (skip_existing=True).
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from .folder_detection import ct_candidates_for_segmentation, read_ct_image, check_ct_image
from .segmentation_sources import discover_moose_models


def _write_information_json(run_dir, models_info, ct_source):
    info = {
        "created_by": "SUV extractor (moosez python API)",
        "source_ct": str(ct_source["path"]),
        "timestamp_stop": datetime.now().strftime("%d.%m.%Y %H:%M"),
        "models": models_info,
    }
    try:
        from moosez import system
        info["moose_version"] = getattr(system, "MOOSE_VERSION", None)
    except Exception:
        pass
    with open(run_dir / "MOOSE_information.json", "w") as f:
        json.dump(info, f, indent=4)


def _run_moose_once(moose, patient_path, ct, models, accelerator):
    """One MOOSE run with one CT; removes its run folder on failure."""
    import shutil
    import time
    run_dir = patient_path / f"moosez-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    while run_dir.exists():                        # retry within the same second
        time.sleep(1)
        run_dir = patient_path / f"moosez-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    seg_dir = run_dir / "segmentations"
    seg_dir.mkdir(parents=True)
    (run_dir / "stats").mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            if ct["kind"] == "nifti":
                ct_path = str(ct["path"])
            else:
                # moosez needs a NIfTI path to reorient correctly to RAS
                import SimpleITK as sitk
                ct_path = os.path.join(tmp, "CT.nii.gz")
                sitk.WriteImage(read_ct_image(ct), ct_path)
            print(f"Running MOOSE {models} on {os.path.relpath(ct['path'], patient_path)} ...")
            seg_files, used_models = moose(ct_path, models, str(seg_dir), accelerator)
        if not seg_files:
            raise RuntimeError("MOOSE produced no segmentation")
        import SimpleITK as sitk
        if not any(sitk.GetArrayFromImage(sitk.ReadImage(str(f))).any() for f in seg_files):
            raise RuntimeError("MOOSE segmentations are all empty")
    except Exception:
        shutil.rmtree(run_dir, ignore_errors=True)   # no half-finished run folders
        raise
    return seg_files, used_models, run_dir, seg_dir


def moose_segment_patient(patient_path, models, ct_name_hints=None, accelerator=None,
                          skip_existing=True):
    """
    Run the given MOOSE models on one patient. Returns the list of models run.
    Raises on failure (caught by moose_segment_images).
    """
    from moosez import moose

    patient_path = Path(patient_path)
    models = [m.lower() for m in models]

    if skip_existing:
        existing = discover_moose_models(patient_path, verbose=False)
        todo = [m for m in models if m not in existing]
        if not todo:
            print(f"✅ {patient_path.name}: all MOOSE models already present - skipped")
            return []
        if len(todo) < len(models):
            print(f"   already present: {[m for m in models if m in existing]}")
        models = todo

    candidates = ct_candidates_for_segmentation(patient_path, ct_name_hints)
    if not candidates:
        raise FileNotFoundError("no CT found")

    # try the CT candidates in order (e.g. faulty AC_CT DICOMs -> CT NIfTI)
    errors = []
    for ct in candidates:
        rel = os.path.relpath(ct["path"], patient_path)
        try:
            ok, reason = check_ct_image(ct)
            if not ok:
                raise ValueError(f"CT not usable: {reason}")
            seg_files, used_models, run_dir, seg_dir = _run_moose_once(
                moose, patient_path, ct, models, accelerator)
            break
        except Exception as e:
            errors.append(f"{rel} ({ct['kind']}): {e}")
            print(f"❌ MOOSE failed with CT {rel}: {e}")
    else:
        raise RuntimeError("MOOSE failed with every CT: " + " | ".join(errors))
    if errors:
        from .issue_log import log_issue
        log_issue(patient_path.name, "CT",
                  f"MOOSE failed with {'; '.join(errors)} -> segmented with "
                  f"{os.path.relpath(ct['path'], patient_path)}")

    models_info = {}
    for seg_file, model in zip(seg_files, used_models):
        model.organ_indices_to_json(str(seg_dir))     # label value -> name, as the CLI does
        models_info[model.model_identifier] = {"segmentation_file": str(seg_file),
                                               "model_directory": getattr(model, "directory", None)}
    _write_information_json(run_dir, models_info, ct)
    print(f"✅ {patient_path.name}: MOOSE output in {run_dir.name}")
    return list(models_info)


def moose_segment_images(data_dir, models, ct_name_hints=None, accelerator=None,
                         skip_existing=True):
    """
    Run MOOSE for every patient folder in data_dir. Failed patients are
    written to <data_dir>/moose_failed_folders.txt.
    """
    data_dir = Path(data_dir)
    failed = []
    patients = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for i, patient_path in enumerate(patients, 1):
        print(f"\n[{i}/{len(patients)}] MOOSE: {patient_path.name}")
        try:
            moose_segment_patient(patient_path, models, ct_name_hints=ct_name_hints,
                                  accelerator=accelerator, skip_existing=skip_existing)
        except Exception as e:
            print(f"❌ MOOSE failed for {patient_path.name}: {e}")
            failed.append(f"{patient_path.name} -> {e}")

    if failed:
        failure_file = data_dir / "moose_failed_folders.txt"
        with open(failure_file, "w") as f:
            f.write("MOOSE failed for the following folders:\n\n")
            f.write("\n".join(failed) + "\n")
        print(f"\nFailure log saved to: {failure_file}")
    print("MOOSE segmentation completed.")