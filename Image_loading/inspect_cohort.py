"""
Dry run: which PET series, CT and segmentations would be used per patient,
and which organs can be found. No images are loaded except DICOM headers.

    python inspect_cohort.py [--data-dir <cohort folder>]     (default: DATA_DIR in config.py)
"""
from pathlib import Path

import pandas as pd

import config as cfg
from Analysis.organ_definitions import organ_config
from Image_loading.folder_detection import find_pet_series, find_ct_source
from Image_loading.segmentation_sources import discover_moose_models, normalise_moose_name, seg_name

# Same organs and combinations as the extraction (Analysis/organ_definitions.py)
ORGANS_TS, COMBINED_TS = organ_config("totalsegmentator")
ORGANS_MOOSE, COMBINED_MOOSE = organ_config("moose")
ORGANS = list(dict.fromkeys(ORGANS_TS + ORGANS_MOOSE + list(COMBINED_TS) + list(COMBINED_MOOSE)
                            + [cfg.REFERENCE_ORGAN, cfg.REFERENCE_VERTEBRA]))


def available(names, organ, combined):
    """'exact', 'combined' (all parts present), 'partial' or '' - exact names only"""
    names = {seg_name(n) for n in names}
    if organ in combined:
        present = [c.lower() in names for c in combined[organ]]
        return "combined" if all(present) else ("partial" if any(present) else "")
    return "exact" if organ.lower() in names else ""


def main(data_dir, output_dir):
    data_dir = Path(data_dir)
    rows = []
    for patient in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        print(f"\n===== {patient.name}")
        pet = find_pet_series(patient, name_hints=cfg.PET_NAME_HINTS)
        ct = find_ct_source(patient, name_hints=cfg.CT_NAME_HINTS)
        row = {"patient": patient.name,
               "pet": pet["folder"].relative_to(patient).as_posix() if pet else None,
               "pet_slices": len(pet["files"]) if pet else None,
               "pet_units": str(pet["header"].get("Units", "")) if pet else None,
               "ct": Path(ct["path"]).relative_to(patient).as_posix() if ct else None}

        ts = patient / "CT_segmentation"
        ts_names = [f.name for f in ts.iterdir()] if ts.is_dir() else []
        models = discover_moose_models(patient)
        moose_names = [normalise_moose_name(n) for m in models.values() for n in m["labels"].values()]
        row["ts_regions"] = len(ts_names)
        row["moose_models"] = ",".join(models)
        for organ in ORGANS:
            row[f"TS:{organ}"] = available(ts_names, organ, COMBINED_TS)
            row[f"MOOSE:{organ}"] = available(moose_names, organ, COMBINED_MOOSE)
        rows.append(row)

        if models:
            print("MOOSE regions:", ", ".join(sorted(set(moose_names))))

    df = pd.DataFrame(rows)
    out_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
    out = out_dir / f"{cohort}_inspection.csv"
    df.to_csv(out, index=False, sep=";")
    print(f"\nSaved {out}")

    print(f"\nCohort: {data_dir.resolve()}\nOrgans found (patients):")
    for organ in ORGANS:
        ts_n = df[f"TS:{organ}"].isin(["exact", "combined"]).sum()
        mo_n = df[f"MOOSE:{organ}"].isin(["exact", "combined"]).sum()
        partial = [t for t in ("TS", "MOOSE") if (df[f"{t}:{organ}"] == "partial").any()]
        note = f"  ⚠️ incomplete combination in {', '.join(partial)}" if partial else ""
        print(f"  {organ:<22} TS {ts_n:>3}/{len(df)}   MOOSE {mo_n:>3}/{len(df)}{note}")


if __name__ == "__main__":
    data_dir, output_dir, _ = cfg.parse_args("Dry run: PET/CT detection and organ availability per patient")
    main(data_dir, output_dir)