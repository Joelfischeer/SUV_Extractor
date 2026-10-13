"""
Segmentation sources: give binary organ masks resampled to the PET grid,
independent of how the segmentation tool stores them.

TotalSegmentator : <patient>/CT_segmentation/<region>.nii.gz, one binary file
                   per region.
MOOSE            : <patient>/moosez-YYYY-MM-DD-HH-MM-SS/segmentations/
                   <prefix>segmentation_<ct>.nii.gz, one MULTI-LABEL file per
                   model (label 1, 2, 3, ...), with the label names in
                   <prefix>organ_indices.json (fallback: stats/*_volume.csv).
                   Several moosez-* folders (different runs/models) are merged;
                   per model the newest run is used.

Masks are produced lazily: a file is only read and resampled when one of its
regions is requested, and a MOOSE multi-label file is resampled to the PET grid
only once (nearest neighbour), then split per label. Splitting after resampling
gives exactly the same masks as resampling every binary mask on its own,
because nearest-neighbour resampling only copies voxel values.

Region names are normalised to the TotalSegmentator convention, so the same
organ names work for both tools (MOOSE 'vertebra_L2' -> 'vertebrae_L2').
"""

import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def seg_name(key):
    """'heart_atrium_left.nii.gz' -> 'heart_atrium_left' (lowercase, extension removed)."""
    k = str(key).lower()
    for ext in (".nii.gz", ".nii", ".mha", ".nrrd"):
        if k.endswith(ext):
            return k[: -len(ext)]
    return k


def match_segmentation(names, organ, verbose=False):
    """
    Exact name match only (case-insensitive, file extension ignored):
    'heart' matches 'heart' / 'heart.nii.gz', never 'heart_atrium_left'.
    Returns None if there is no exact match.
    """
    organ_l = organ.lower()
    return next((k for k in names if seg_name(k) == organ_l), None)


# ---------------------------------------------------------------------------
# MOOSE name normalisation
# ---------------------------------------------------------------------------
# MOOSE region name -> project name (TotalSegmentator convention)
MOOSE_NAME_ALIASES = {
    # vertebrae: MOOSE 'vertebra_L2', TotalSegmentator 'vertebrae_L2'
    **{f"vertebra_{p}{i}": f"vertebrae_{p}{i}" for p, n in (("C", 7), ("T", 12), ("L", 6))
       for i in range(1, n + 1)},
}


def normalise_moose_name(name):
    n = re.sub(r"[\s\-]+", "_", str(name).strip()).strip("_")
    # keep vertebra level upper case (L2), everything else lower case
    m = re.match(r"(?i)^vertebra[e]?_([CTL])(\d+)$", n)
    if m:
        n = f"vertebra_{m.group(1).upper()}{m.group(2)}"
    else:
        n = n.lower()
    return MOOSE_NAME_ALIASES.get(n, n)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
class _Source:
    backend = None

    def __init__(self, pet_reference):
        self.pet_reference = pet_reference

    def names(self):
        raise NotImplementedError

    def mask(self, name):
        """uint8 array (z, y, x) on the PET grid, or None."""
        raise NotImplementedError

    def _resample(self, img, pixel_type):
        # LPS first, as in the original pipeline: physically identical, but it
        # fixes the nearest-neighbour rounding at exact voxel ties, so masks
        # are voxel-identical to earlier results.
        img = sitk.DICOMOrient(img, "LPS")
        aligned = sitk.Resample(img, self.pet_reference, sitk.Transform(),
                                sitk.sitkNearestNeighbor, 0, pixel_type)
        return sitk.GetArrayFromImage(aligned)

    def release(self):
        pass


class TotalSegmentatorSource(_Source):
    """One binary NIfTI per region in <patient>/CT_segmentation."""
    backend = "totalsegmentator"

    def __init__(self, segmentation_folder, pet_reference):
        super().__init__(pet_reference)
        self.folder = Path(segmentation_folder)
        self.files = {seg_name(f.name): f for f in sorted(self.folder.iterdir())
                      if f.is_file() and f.name.lower().endswith((".nii.gz", ".nii", ".mha", ".nrrd"))}

    def names(self):
        return list(self.files)

    def mask(self, name):
        path = self.files.get(name)
        if path is None:
            return None
        arr = self._resample(sitk.ReadImage(str(path)), sitk.sitkUInt8)
        return (arr > 0).astype(np.uint8)

    def describe(self):
        return f"TotalSegmentator: {self.folder.name}/ ({len(self.files)} regions)"


def _moose_run_time(run_dir):
    """moosez-2025-03-04-10-22-33 -> sortable string; fallback mtime."""
    m = re.search(r"(\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2})", run_dir.name)
    return m.group(1) if m else f"{run_dir.stat().st_mtime:020.6f}"


def _read_organ_indices_json(path):
    """{"organ_indices": {"1": {"name": "liver", ...}, ...}} -> {1: 'liver'}"""
    with open(path) as f:
        data = json.load(f)
    data = data.get("organ_indices", data)
    out = {}
    for k, v in data.items():
        name = v.get("name") if isinstance(v, dict) else v
        if str(k).isdigit() and name:
            out[int(k)] = str(name)
    return out


def _read_labels_from_stats(stats_dir, prefix):
    """Fallback: stats/<prefix><patient>_volume.csv, index = label, column 'Regions-Present'."""
    import pandas as pd
    for csv in sorted(Path(stats_dir).glob(f"{prefix}*.csv")):
        try:
            df = pd.read_csv(csv, index_col=0)
        except Exception:
            continue
        if "Regions-Present" in df.columns:
            return {int(i): str(n) for i, n in df["Regions-Present"].items()
                    if str(i).isdigit() and int(i) != 0}
    return {}


def discover_moose_models(patient_path, verbose=True):
    """
    Find all MOOSE multi-label segmentations of a patient.

    Returns {model_id: {"file": Path, "labels": {int: raw_name}, "run": dirname}},
    newest run per model. model_id is the MOOSE model identifier
    (e.g. 'clin_ct_organs'), derived from MOOSE_information.json or the file
    prefix ('clin_CT_organs_segmentation_AC_CT.nii.gz' -> 'clin_ct_organs').
    """
    patient_path = Path(patient_path)
    runs = sorted((d for d in patient_path.iterdir() if d.is_dir() and d.name.lower().startswith("moosez-")),
                  key=_moose_run_time, reverse=True)       # newest first
    models = {}

    for run in runs:
        seg_dir = run / "segmentations"
        if not seg_dir.is_dir():
            continue

        # model ids from MOOSE_information.json (file name -> model id)
        file_to_model = {}
        info_path = run / "MOOSE_information.json"
        if info_path.exists():
            try:
                with open(info_path) as f:
                    info = json.load(f)
                for model_id, minfo in (info.get("models") or {}).items():
                    seg_file = (minfo or {}).get("segmentation_file")
                    if seg_file:
                        # absolute paths may come from another machine -> file name only
                        file_to_model[Path(str(seg_file).replace("\\", "/")).name] = model_id
            except Exception as e:
                print(f"⚠️ Could not read {info_path}: {e}")

        for seg_file in sorted(seg_dir.glob("*.nii*")):
            m = re.match(r"^(.*?_)segmentation_.*\.nii(\.gz)?$", seg_file.name)
            if not m:
                continue
            prefix = m.group(1)                                   # 'clin_CT_organs_'
            model_id = file_to_model.get(seg_file.name, prefix.rstrip("_").lower())
            if model_id in models:
                continue                                          # newer run already used

            labels = {}
            json_path = seg_dir / f"{prefix}organ_indices.json"
            if json_path.exists():
                labels = _read_organ_indices_json(json_path)
            if not labels and (run / "stats").is_dir():
                labels = _read_labels_from_stats(run / "stats", prefix)
            if not labels:
                print(f"⚠️ {run.name}/{seg_file.name}: no label names found - skipped")
                continue

            models[model_id] = {"file": seg_file, "labels": labels, "run": run.name}

    if verbose:
        if not models:
            print(f"⚠️ No MOOSE segmentations found in {patient_path}")
        for model_id, m in models.items():
            print(f"MOOSE {model_id:<28} {len(m['labels']):>3} labels  ({m['run']})")
    return models


class MooseSource(_Source):
    """Multi-label MOOSE segmentations, merged over models and runs."""
    backend = "moose"

    def __init__(self, patient_path, pet_reference, model_priority=None, verbose=True):
        super().__init__(pet_reference)
        self.models = discover_moose_models(patient_path, verbose=verbose)
        priority = [m.lower() for m in (model_priority or [])]
        order = sorted(self.models, key=lambda m: priority.index(m) if m in priority else len(priority))

        # region name -> (model_id, label value); higher priority model wins
        self.regions = {}
        self.duplicates = []
        for model_id in order:
            for label, raw in sorted(self.models[model_id]["labels"].items()):
                name = normalise_moose_name(raw)
                if name in self.regions:
                    self.duplicates.append((name, self.regions[name][0], model_id))
                    continue
                self.regions[name] = (model_id, label)
        if self.duplicates and verbose:
            dup = ", ".join(f"{n} ({a} over {b})" for n, a, b in self.duplicates)
            print(f"     regions in several models, used: {dup}")
        self._pet_label_maps = {}

    def names(self):
        return list(self.regions)

    def _label_map(self, model_id):
        if model_id not in self._pet_label_maps:
            img = sitk.ReadImage(str(self.models[model_id]["file"]))
            self._pet_label_maps[model_id] = self._resample(img, sitk.sitkUInt16)
        return self._pet_label_maps[model_id]

    def mask(self, name):
        if name not in self.regions:
            return None
        model_id, label = self.regions[name]
        return (self._label_map(model_id) == label).astype(np.uint8)

    def model_of(self, name):
        return self.regions.get(name, (None, None))[0]

    def release(self):
        self._pet_label_maps.clear()

    def describe(self):
        return f"MOOSE: {len(self.models)} models, {len(self.regions)} regions"


class CombinedMasks:
    """
    Wraps a source and adds the combination logic (OR of several regions).
    Combined names take precedence over source names, as before.
    """

    def __init__(self, source, combination_logic=None):
        self.source = source
        self.combination_logic = combination_logic or {}

    def names(self):
        return list(dict.fromkeys(list(self.combination_logic) + self.source.names()))

    def resolve(self, organ, verbose=True):
        """Name under which `organ` is available, or None."""
        if organ in self.combination_logic:
            return organ
        return match_segmentation(self.source.names(), organ, verbose=verbose)

    def mask(self, name):
        if name in self.combination_logic:
            combined, used = None, []
            for comp in self.combination_logic[name]:
                key = match_segmentation(self.source.names(), comp)
                if key is None:
                    continue
                part = self.source.mask(key)
                combined = part if combined is None else np.maximum(combined, part)
                used.append(key)
            return combined
        return self.source.mask(name)


def get_segmentation_source(patient_path, backend, pet_reference, moose_model_priority=None,
                            ts_folder_name="CT_segmentation", verbose=True):
    """Factory: 'totalsegmentator' or 'moose'. Returns None if nothing is found."""
    backend = backend.lower()
    patient_path = Path(patient_path)
    if backend == "totalsegmentator":
        folder = patient_path / ts_folder_name
        if not folder.is_dir():
            # tolerate different capitalisation (CT_Segmentation / CT_segmentation)
            folder = next((d for d in patient_path.iterdir()
                           if d.is_dir() and d.name.lower() == ts_folder_name.lower()), None)
        if folder is None:
            print("⚠️ No TotalSegmentator segmentation folder found.")
            return None
        src = TotalSegmentatorSource(folder, pet_reference)
    elif backend == "moose":
        src = MooseSource(patient_path, pet_reference, model_priority=moose_model_priority, verbose=verbose)
        if not src.regions:
            return None
    else:
        raise ValueError(f"Unknown segmentation backend '{backend}' (use 'totalsegmentator' or 'moose')")
    if verbose:
        print(src.describe())
    return src