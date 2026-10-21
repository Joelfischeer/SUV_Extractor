"""
Run CADS (https://github.com/murong-xu/CADS) on the CT of a patient.

Output per patient:

    <patient>/cads_segmentation/
        <ct>_part_551.nii.gz ... <ct>_part_559.nii.gz   (multi-label, one per task)
        cads_labelmap.json                              (task -> label value -> name)
        CADS_information.json                           (tasks, model variant, CT used)

Two ways to call CADS:
  1. python API, if the 'cads' package is installed in this environment
  2. command line in a separate environment (recommended by the CADS authors;
     CADS pins nnunetv2==2.5.2, which can conflict with TotalSegmentator/MOOSE):
         CADS_COMMAND = ["conda", "run", "-n", "CADS_env", "python",
                         "/path/to/CADS/cads/scripts/predict_images.py"]
     in config.py. The command line accepts one task (or 'all') per call, so
     it is called once per task.

Tasks 557/558 need 553 (and 558 needs 552); CADS adds these automatically.
"""
import json
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .folder_detection import ct_candidates_for_segmentation, read_ct_image, check_ct_image
from .cads_labelmaps import CADS_LABELMAPS

CADS_FOLDER = "cads_segmentation"
ALL_TASKS = (551, 552, 553, 554, 555, 556, 557, 558, 559)


def _cads_segmentation_installed():
    """
    True only for the CADS segmentation tool (github.com/murong-xu/CADS). The
    PyPI package 'cads' is an unrelated Copernicus download tool, so the check
    looks for the tool's own inference module, not just a module called cads.
    """
    import importlib.util
    try:
        return importlib.util.find_spec("cads.utils.inference") is not None
    except (ImportError, ValueError):
        return False


def _run_cads_api(ct_path, out_dir, tasks, license, use_cpu):
    from cads.utils.libs import setup_nnunet_env, get_model_weights_dir, check_or_download_model_weights
    setup_nnunet_env(license)
    from cads.utils.inference import predict
    for task in tasks:
        check_or_download_model_weights(task, license)
    if any(t in tasks for t in (557, 558)) and 553 not in tasks:
        check_or_download_model_weights(553, license)
    if 558 in tasks and 552 not in tasks:
        check_or_download_model_weights(552, license)
    predict([str(ct_path)], str(out_dir), get_model_weights_dir(license), sorted(tasks), folds="all",
            use_cpu=use_cpu, preprocess_cads=True, postprocess_cads=True,
            save_all_combined_seg=False, snapshot=False, save_separate_targets=False, mode="auto")


def _run_cads_command(command, ct_path, out_dir, tasks, license, use_cpu):
    task_args = ["all"] if set(tasks) == set(ALL_TASKS) else [str(t) for t in sorted(tasks)]
    for task in task_args:
        cmd = list(command) + ["-in", str(ct_path), "-out", str(out_dir), "-task", task,
                               "-license", license] + (["--cpu"] if use_cpu else [])
        print("   $", " ".join(cmd))
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode != 0:
            raise RuntimeError(f"CADS command failed (task {task}): {done.stderr.strip()[-500:]}")


def _labelmaps():
    """Label maps of the installed CADS version, else the built-in copy."""
    try:
        from cads.dataset_utils.bodyparts_labelmaps import map_taskid_to_labelmaps
        return {t: {k: v for k, v in lm.items() if k != 0} for t, lm in map_taskid_to_labelmaps.items()}, "installed CADS"
    except Exception:
        return CADS_LABELMAPS, "built-in copy (CADS 1.0.3)"


def _run_cads_once(patient_path, ct, tasks, license, use_cpu, command):
    """One CADS run with one CT; returns the produced part files (in a temp dir)."""
    import SimpleITK as sitk
    tmp = Path(tempfile.mkdtemp(prefix="cads_"))
    try:
        # CADS reads *.nii.gz and names the case after the file
        if ct["kind"] == "nifti" and str(ct["path"]).endswith(".nii.gz"):
            ct_path = Path(ct["path"])
        else:
            ct_path = tmp / "CT.nii.gz"
            sitk.WriteImage(read_ct_image(ct), str(ct_path))
        out = tmp / "out"
        out.mkdir()
        print(f"Running CADS tasks {sorted(tasks)} ({license}) on "
              f"{os.path.relpath(ct['path'], patient_path)} ...")
        if command:
            _run_cads_command(command, ct_path, out, tasks, license, use_cpu)
        else:
            _run_cads_api(ct_path, out, tasks, license, use_cpu)
        parts = sorted(out.rglob("*_part_55*.nii.gz"))
        produced = {int(p.name.split("_part_")[-1][:3]): p for p in parts}
        missing = [t for t in tasks if t not in produced]
        if missing:
            raise RuntimeError(f"CADS produced no output for tasks {missing}")
        if not any(sitk.GetArrayFromImage(sitk.ReadImage(str(p))).any() for p in produced.values()):
            raise RuntimeError("CADS segmentations are all empty")
        return produced, tmp
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def cads_segment_patient(patient_path, tasks, ct_name_hints=None, license="research",
                         use_cpu=False, command=None):
    """
    Run the given CADS tasks for one patient (CT candidates tried in order,
    e.g. faulty AC_CT DICOMs -> CT NIfTI). Raises if every CT fails.
    """
    if not command and not _cads_segmentation_installed():
        raise RuntimeError("CADS (murong-xu/CADS) is not installed in this environment - set "
                           "CADS_COMMAND in config.py to run it from its own environment. "
                           "Note: the PyPI package 'cads' is an unrelated Copernicus tool.")
    patient_path = Path(patient_path)
    candidates = ct_candidates_for_segmentation(patient_path, ct_name_hints)
    if not candidates:
        raise FileNotFoundError("no CT found")

    errors = []
    for ct in candidates:
        rel = os.path.relpath(ct["path"], patient_path)
        try:
            ok, reason = check_ct_image(ct)
            if not ok:
                raise ValueError(f"CT not usable: {reason}")
            produced, tmp = _run_cads_once(patient_path, ct, tasks, license, use_cpu, command)
            break
        except Exception as e:
            errors.append(f"{rel} ({ct['kind']}): {e}")
            print(f"❌ CADS failed with CT {rel}: {e}")
    else:
        raise RuntimeError("CADS failed with every CT: " + " | ".join(errors))
    if errors:
        from .issue_log import log_issue
        log_issue(patient_path.name, "CT", f"CADS failed with {'; '.join(errors)} -> segmented with {rel}")

    out_dir = patient_path / CADS_FOLDER
    out_dir.mkdir(exist_ok=True)
    stem = Path(ct["path"]).name.replace(".nii.gz", "").replace(".nii", "")
    try:
        for task, f in produced.items():
            for old in out_dir.glob(f"*_part_{task}.nii*"):      # replace an older result of the task
                old.unlink()
            shutil.move(str(f), out_dir / f"{stem}_part_{task}.nii.gz")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    maps, maps_source = _labelmaps()
    with open(out_dir / "cads_labelmap.json", "w") as fh:
        json.dump({str(t): {str(k): v for k, v in lm.items()} for t, lm in maps.items()}, fh, indent=2)
    info_path = out_dir / "CADS_information.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {"runs": []}
    info["runs"].append({"date": datetime.now().strftime("%Y-%m-%d %H:%M"), "tasks": sorted(produced),
                         "license": license, "source_ct": str(ct["path"]), "label_map": maps_source,
                         "via": "command" if command else "python API"})
    info_path.write_text(json.dumps(info, indent=2))
    print(f"✅ {patient_path.name}: CADS tasks {sorted(produced)} in {CADS_FOLDER}/")
    return sorted(produced)