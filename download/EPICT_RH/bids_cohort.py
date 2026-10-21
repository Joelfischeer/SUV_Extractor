"""
PET from BIDS cohorts (e.g. Multimodal-HC): NIfTI in Bq/mL + JSON sidecar,
no DICOM.

    <cohort>/participants.tsv                     weight, height, age, sex, ...
    <cohort>/derivatives/readouts/metadata.csv    'SUV Denominator [Bq/mL]' per subject
    <cohort>/sub-000/ses-*/pet/*_rec-acstatPSF_pet.nii.gz (+ .json)

SUV = Bq/mL x factor, factor from two sources:
  1. the cohort's own SUV denominator (metadata.csv)        -> factor = 1 / denominator
  2. sidecar + weight: weight [g] / dose at the decay-correction reference
     (sidecar ImageDecayCorrectionTime relative to InjectionStart; 0 = images
     corrected to the injection -> no decay of the dose)
Source 1 is used if available, source 2 checks it (difference > 2 % is logged).
Without a denominator, source 2 alone is used.
"""
import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DOSE_UNITS = {"bq": 1.0, "kbq": 1e3, "mbq": 1e6, "gbq": 1e9, "mci": 3.7e7, "uci": 3.7e4, "ci": 3.7e10}


def subject_id(value):
    """'sub-000', '000', 0 -> 'sub-000'"""
    s = str(value).strip()
    if s.lower().startswith("sub-"):
        return "sub-" + s[4:]
    if re.fullmatch(r"\d+(\.0)?", s):
        return f"sub-{int(float(s)):03d}"
    return f"sub-{s}"


def find_bids_pet(patient_path, rec_hints=("acstatPSF", "acstatOSEM")):
    """
    Static attenuation-corrected PET of a BIDS subject: (nifti Path, sidecar dict)
    or None. Dynamic ('dyn') and non-attenuation-corrected ('nac') reconstructions
    and head-only reconstructions are skipped; rec_hints give the preference.
    """
    candidates = []
    for nii in Path(patient_path).rglob("*_pet.nii*"):
        if "derivatives" in nii.parts:
            continue
        rec = re.search(r"_rec-([A-Za-z0-9]+)_", nii.name)
        rec = rec.group(1) if rec else ""
        if re.search(r"dyn|^nac|head", rec, re.I):
            continue
        sidecar_path = Path(re.sub(r"\.nii(\.gz)?$", ".json", str(nii)))
        if not sidecar_path.exists():
            continue
        with open(sidecar_path) as f:
            sidecar = json.load(f)
        if str(sidecar.get("Units", "")).lower().replace(" ", "") not in ("bq/ml", "bqml"):
            continue
        rank = next((i for i, h in enumerate(rec_hints) if h.lower() == rec.lower()), len(rec_hints))
        candidates.append((rank, nii.name, nii, sidecar))
    if not candidates:
        return None
    candidates.sort(key=lambda c: (c[0], c[1]))
    return candidates[0][2], candidates[0][3]


def _find_col(df, *patterns, exclude=()):
    for col in df.columns:
        name = str(col).lower()
        if all(re.search(p, name) for p in patterns) and not any(re.search(e, name) for e in exclude):
            return col
    return None


def _num(series):
    """Numbers with decimal point or decimal comma ('1,84' in participants.tsv)."""
    text = series.astype(str).str.strip()
    text = text.where(~text.str.fullmatch(r"-?\d+,\d+"), text.str.replace(",", ".", regex=False))
    return pd.to_numeric(text, errors="coerce")


def _read_table(path):
    sep = "\t" if str(path).endswith(".tsv") else None
    return pd.read_csv(path, sep=sep, engine="python")


@lru_cache(maxsize=8)
def load_clinical(cohort_dir):
    """
    One row per subject (index 'sub-XXX') with weight_kg, height_m, age, sex,
    suv_denominator_bqml - from participants.tsv and
    derivatives/readouts/metadata.csv, whatever is available.
    """
    cohort_dir = Path(cohort_dir)
    tables = []
    for path in (cohort_dir / "participants.tsv",
                 *sorted((cohort_dir / "derivatives" / "readouts").glob("metadata*"))):
        if not path.exists():
            continue
        try:
            df = _read_table(path)
        except Exception as e:
            print(f"⚠️ Could not read {path}: {e}")
            continue
        id_col = _find_col(df, r"participant_id") or _find_col(df, r"^subject") or _find_col(df, r"sub")
        if id_col is None:
            continue
        df = df.drop_duplicates(id_col).set_index(df[id_col].map(subject_id))
        out = pd.DataFrame(index=df.index)
        weight = _find_col(df, r"weight")
        height = _find_col(df, r"height") or _find_col(df, r"^size")
        age, sex = _find_col(df, r"^age"), _find_col(df, r"^sex|gender")
        denom = _find_col(df, r"suv", r"denominator")
        if weight is not None:
            w = _num(df[weight])
            out["weight_kg"] = np.where(w > 500, w / 1000, w)            # grams -> kg
        if height is not None:
            h = _num(df[height])
            out["height_m"] = np.where(h > 3, h / 100, h)               # cm -> m
        if age is not None:
            out["age"] = _num(df[age])
        if sex is not None:
            out["sex"] = df[sex]
        if denom is not None:
            out["suv_denominator_bqml"] = _num(df[denom])
        tables.append(out)
    if not tables:
        return pd.DataFrame()
    merged = tables[0]
    for t in tables[1:]:
        merged = merged.combine_first(t)          # participants.tsv first, then metadata
    return merged


def _dose_bq(sidecar):
    dose = sidecar.get("InjectedRadioactivity")
    unit = str(sidecar.get("InjectedRadioactivityUnits", "MBq")).lower().replace("µ", "u")
    try:
        return float(dose) * DOSE_UNITS[unit]
    except (TypeError, ValueError, KeyError):
        return None


def bids_suv_factor(subject, sidecar, cohort_dir, tolerance=0.02):
    """
    SUV factor (multiply Bq/mL) for one subject.
    Returns (factor or None, info dict, warning text or '').
    """
    clinical = load_clinical(str(cohort_dir))
    row = clinical.loc[subject] if subject in clinical.index else pd.Series(dtype=object)
    info = {"units": "BQML", "decay_correction": "ImageDecayCorrectionTime "
                                                  f"{sidecar.get('ImageDecayCorrectionTime')}"}

    # uptake time: frame start relative to injection (BIDS times relative to TimeZero)
    inj = float(sidecar.get("InjectionStart", 0) or 0)
    starts = sidecar.get("FrameTimesStart") or [None]
    if starts[0] is not None:
        info["uptake_time_min"] = (float(starts[0]) - inj) / 60

    # source 2: sidecar dose + weight
    own = None
    dose, weight = _dose_bq(sidecar), row.get("weight_kg")
    half_life = float(sidecar.get("RadionuclideHalfLife", 6586.2) or 6586.2)
    if dose and weight is not None and pd.notna(weight) and weight > 0:
        if sidecar.get("ImageDecayCorrected", True):
            t_ref = float(sidecar.get("ImageDecayCorrectionTime", inj) or 0) - inj
        else:     # not decay corrected: activity at the frame start
            t_ref = float(starts[0] or 0) - inj
        dose_ref = dose * math.exp(-math.log(2) / half_life * t_ref)
        own = weight * 1000 / dose_ref
        info.update({"injected_dose_Bq": dose, "weight_kg": float(weight),
                     "decay_reference_after_injection_s": t_ref})

    # source 1: the cohort's SUV denominator
    denom = row.get("suv_denominator_bqml")
    official = 1.0 / denom if denom is not None and pd.notna(denom) and denom > 0 else None

    warning = ""
    if official is not None:
        factor, method = official, "BIDS NIfTI, SUV denominator (metadata.csv)"
        if own is not None and abs(own / official - 1) > tolerance:
            warning = (f"SUV factor from sidecar dose + weight ({own:.4e}) differs "
                       f"{100 * (own / official - 1):+.1f} % from the cohort's SUV denominator "
                       f"({official:.4e}) - denominator used")
    elif own is not None:
        factor, method = own, "BIDS NIfTI, sidecar dose + weight (participants.tsv)"
    else:
        return None, info, ("no SUV factor: neither an SUV denominator (metadata.csv) nor "
                            "dose (sidecar) + weight (participants.tsv) available")
    info.update({"method": method, "suv_factor": factor})
    if own is not None and official is not None:
        info["suv_factor_check"] = own
    return factor, info, warning


def bids_patient_info(subject, cohort_dir):
    """dict weight_kg, height_m, age, sex from the clinical tables (None if missing)."""
    clinical = load_clinical(str(cohort_dir))
    row = clinical.loc[subject] if subject in clinical.index else pd.Series(dtype=object)
    out = {}
    for key in ("weight_kg", "height_m", "age", "sex"):
        v = row.get(key)
        out[key] = None if v is None or (not isinstance(v, str) and pd.isna(v)) else v
    for key in ("weight_kg", "height_m", "age"):
        out[key] = float(out[key]) if out[key] is not None else None
    return out