"""
Check the PET timing of every patient: which time the images are really
decay-corrected to, and whether the SUV conversion uses the right one.

    python check_pet_timing.py --data-dir <cohort folder>

The SUV conversion decays the injected dose to the series start (SeriesTime)
for DecayCorrection START. That is only right if the scanner corrected the
pixel values to that time. The DecayFactor tag (0054,1321) tells what the
scanner did:

    implied reference = frame mid time - ln(DecayFactor) / lambda

and is compared with the injection time, the SeriesTime and the frame start.
Output: <OUTPUT_DIR>/<cohort>_SUV_extraction/<cohort>_pet_timing.csv
"""
import math

import pandas as pd
import pydicom

import config as cfg
from Image_loading.folder_detection import find_pet_series, list_patient_dirs
from Image_loading.Image_Loader import _dicom_time_to_seconds, _series_start_seconds


def _clock(sec):
    if sec is None or not math.isfinite(sec):
        return None
    sec %= 86400
    return f"{int(sec // 3600):02d}:{int(sec % 3600 // 60):02d}:{int(sec % 60):02d}"


def check_patient(patient_path):
    pet = find_pet_series(patient_path, name_hints=cfg.PET_NAME_HINTS, verbose=False)
    if pet is None:
        return {"patient": patient_path.name, "verdict": "no PET found"}
    headers = [pydicom.dcmread(f, stop_before_pixels=True, force=True) for f in pet["files"]]
    h = headers[0]
    rad = h.RadiopharmaceuticalInformationSequence[0]
    half_life = float(rad.RadionuclideHalfLife)
    lam = math.log(2) / half_life

    t_inj = _dicom_time_to_seconds(rad.RadiopharmaceuticalStartTime)
    t_series = _dicom_time_to_seconds(h.SeriesTime) if h.get("SeriesTime") else float("nan")
    acq = [_dicom_time_to_seconds(x.AcquisitionTime) for x in headers if x.get("AcquisitionTime")]
    t_acq = min(acq) if acq else float("nan")
    t_used, used_label = _series_start_seconds(h, headers)

    duration_s = float(h.get("ActualFrameDuration", 0) or 0) / 1000
    decay_factor = float(h.get("DecayFactor", 0) or 0)
    frame_ref_s = float(h.get("FrameReferenceTime", 0) or 0) / 1000

    row = {
        "patient": patient_path.name,
        "pet_series": pet["folder"].name,
        "series_description": str(h.get("SeriesDescription", "")),
        "manufacturer": str(h.get("Manufacturer", "")),
        "decay_correction": str(h.get("DecayCorrection", "")),
        "injection": _clock(t_inj),
        "series_time": _clock(t_series),
        "first_acquisition_time": _clock(t_acq),
        "frame_duration_min": round(duration_s / 60, 2),
        "frame_reference_time_min": round(frame_ref_s / 60, 2),
        "decay_factor": decay_factor or None,
        "uptake_used_min": round(((t_used - t_inj) % 86400) / 60, 1),
        "uptake_to_frame_start_min": round(((t_acq - t_inj) % 86400) / 60, 1) if acq else None,
    }

    if decay_factor > 0 and acq:
        t_mid = t_acq + duration_s / 2
        t_ref = t_mid - math.log(decay_factor) / lam
        row["implied_decay_reference"] = _clock(t_ref)
        candidates = {"injection": t_inj, "SeriesTime": t_series, "frame start": t_acq}
        dist = {k: abs(((t_ref - v + 43200) % 86400) - 43200) / 60
                for k, v in candidates.items() if math.isfinite(v)}
        closest = min(dist, key=dist.get)
        row["implied_reference_is"] = closest
        row["minutes_off"] = round(dist[closest], 1)
        # used reference vs. implied reference
        # dose is decayed to t_used, pixels are corrected to t_ref:
        # SUV_current / SUV_correct = exp(-lambda * (t_ref - t_used))
        delta = ((t_ref - t_used + 43200) % 86400) - 43200
        error_pct = 100 * (math.exp(-lam * delta) - 1)
        row["suv_error_pct"] = round(error_pct, 1)
        row["verdict"] = ("OK - conversion uses the correct reference time" if abs(delta) <= 180
                          else f"WRONG reference ({used_label}) - current SUVs {error_pct:+.1f} %")
    else:
        row["verdict"] = "no DecayFactor - cannot verify"
    return row


def main():
    data_dir, output_dir, _ = cfg.parse_args("Check PET decay-correction timing per patient")
    rows = [check_patient(p) for p in list_patient_dirs(data_dir)]
    df = pd.DataFrame(rows)
    out_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
    out = out_dir / f"{cohort}_pet_timing.csv"
    df.to_csv(out, index=False, sep=";")

    pd.set_option("display.width", 250)
    cols = ["patient", "injection", "series_time", "first_acquisition_time", "decay_factor",
            "uptake_used_min", "uptake_to_frame_start_min", "implied_reference_is", "verdict"]
    print(df[[c for c in cols if c in df]].to_string(index=False))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()