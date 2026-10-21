"""
Download the Multimodal-HC dataset (DEPICT-RH): total-body [18F]FDG PET/CT/MRI
of 100 healthy volunteers, 80 of them currently public (train split).

    GitHub:       https://github.com/DEPICT-RH/Multimodal-HC
    Data (NIfTI): https://huggingface.co/datasets/DEPICT-RH/Multimodal-HC  (CC-BY-4.0)
    Please cite:  Hinge et al., Scientific Data (2026), doi:10.1038/s41597-026-08157-4

The full dataset is 302 GB, 75 % of it dynamic PET. By default the clinical
values and the static images are downloaded - never the dynamic PET
(BIDS layout, kept as is):

    <target>/train/participants.tsv, dataset_description.json, ...  clinical values
    <target>/train/derivatives/readouts/metadata*                    (weight, dose,
                                                                      SUV denominator, ...)
    <target>/train/sub-000/ses-quadra/pet/sub-000_..._rec-acstatPSF_pet.nii.gz (+ .json)
    <target>/train/sub-000/ses-quadra/ct/sub-000_..._rec-br38f_ct.nii.gz      (+ .json)

i.e. the static attenuation-corrected whole-body PET and the CT, about 0.25 GB
per subject. The JSON sidecars (a few kB) are always included: they hold the
injected dose, units (Bq/mL) and decay correction needed for SUVs.

Optional
    --no-clinical            images only
    --with-readouts          all pre-computed readouts (TACs, SUVs per region)
    --with-totalsegmentator  the provided TotalSegmentator segmentations
    --with-mri               the MRI session
    --content no_dynamic     all static PET reconstructions (incl. OSEM, head,
                             non-AC), CT and topogram (~71 GB in total)
    --content all            everything incl. the dynamic PET (~302 GB)

Usage
    python download_multimodal_hc.py --target-dir ../data/Images/Multimodal_HC --dry-run
    python download_multimodal_hc.py --target-dir ../data/Images/Multimodal_HC     # clinical values + static images
    python download_multimodal_hc.py --target-dir ... --n-subjects 5 --with-totalsegmentator
    python download_multimodal_hc.py --target-dir ... --subjects sub-000 sub-001

Requires: pip install "huggingface_hub>=0.25" hf_xet   (hf_xet = fast download backend)
No account or token is needed (public dataset). Downloads resume if interrupted,
and files already present are skipped.
"""
import argparse
import fnmatch
import shutil
from pathlib import Path

REPO_ID = "DEPICT-RH/Multimodal-HC"
REPO_TYPE = "dataset"
SPLIT = "train"
CONTENT_PRESETS = ("static", "no_dynamic", "all")


def _subject_patterns(sub, content, ct_reconstruction, pet_reconstruction, with_mri):
    root = f"{SPLIT}/{sub}"
    if content == "static":
        patterns = [f"{root}/ses-quadra/pet/*rec-{pet_reconstruction}_pet.*",
                    f"{root}/ses-quadra/ct/*rec-{ct_reconstruction}_ct.*"]
        if with_mri:
            patterns.append(f"{root}/ses-vida/*")
        return patterns
    patterns = [f"{root}/ses-quadra/*"]
    if with_mri:
        patterns.append(f"{root}/ses-vida/*")
    return patterns


def build_patterns(subjects, content="static", with_totalsegmentator=False, with_mri=False,
                   with_clinical=True, with_readouts=False, ct_reconstruction="br38f",
                   pet_reconstruction="acstatPSF", root_files=()):
    """
    allow/ignore patterns (fnmatch, as used by huggingface_hub) for the
    selection. subjects=None means all subjects.
    Note: in fnmatch '*' also matches '/', so 'train/*.json' would select every
    JSON sidecar of every subject - the dataset-level files (participants.tsv,
    dataset_description.json, ...) are therefore passed by exact name.
    """
    if content not in CONTENT_PRESETS:
        raise ValueError(f"content must be one of {CONTENT_PRESETS}")
    subs = subjects or ["sub-*"]
    allow = []
    if with_clinical:      # dataset/participant info + clinical metadata (small)
        allow += list(root_files) + [f"{SPLIT}/phenotype/*",
                                     f"{SPLIT}/derivatives/readouts/metadata*"]
    for sub in subs:
        allow += _subject_patterns(sub, content, ct_reconstruction, pet_reconstruction,
                                   with_mri or content == "all")
        if with_totalsegmentator:
            allow.append(f"{SPLIT}/derivatives/totalsegmentator/{sub}/*")
    if with_readouts:      # all pre-computed readouts (TACs, SUVs per region)
        allow.append(f"{SPLIT}/derivatives/readouts/*")
    if content == "all" and subjects is None:
        allow = [f"{SPLIT}/*"]
    ignore = ["*acdynPSF*"] if content in ("static", "no_dynamic") else []
    return allow, ignore


def _selected(path, allow, ignore):
    return (any(fnmatch.fnmatch(path, p) for p in allow)
            and not any(fnmatch.fnmatch(path, p) for p in ignore))


def _split_root(api):
    """(subject folders, dataset-level file paths) of the split."""
    items = list(api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=SPLIT))
    subjects = sorted(Path(i.path).name for i in items
                      if Path(i.path).name.startswith("sub-") and not hasattr(i, "size"))
    root_files = sorted(i.path for i in items if hasattr(i, "size"))
    return subjects, root_files


def list_subjects(api=None):
    """Subject folders available in the public split."""
    from huggingface_hub import HfApi
    return _split_root(api or HfApi())[0]


def plan_download(allow, ignore, api=None):
    """Files that match the selection and their total size (bytes)."""
    from huggingface_hub import HfApi
    api = api or HfApi()
    files = [f for f in api.list_repo_tree(REPO_ID, repo_type=REPO_TYPE, path_in_repo=SPLIT,
                                           recursive=True)
             if hasattr(f, "size") and _selected(f.path, allow, ignore)]
    return files, sum(f.size or 0 for f in files)


def download_multimodal_hc(target_dir, subjects=None, n_subjects=None, content="static",
                           with_totalsegmentator=False, with_mri=False, with_clinical=True,
                           with_readouts=False,
                           ct_reconstruction="br38f", pet_reconstruction="acstatPSF",
                           dry_run=False, max_workers=4):
    """
    Download (a selection of) the Multimodal-HC dataset into target_dir.

    Parameters
    ----------
    target_dir : folder; the data ends up in <target_dir>/train/sub-XXX/...
    subjects : list like ["sub-000", "sub-001"]; None = all available
    n_subjects : take only the first n available subjects (ignored if subjects given)
    content : "static" (default: static PET + CT), "no_dynamic" (all static
              reconstructions) or "all" (incl. dynamic PET)
    with_totalsegmentator : also the provided TotalSegmentator segmentations
    with_mri : also the MRI session (ses-vida)
    with_clinical : participants.tsv, dataset-level files, phenotype/ and
                    derivatives/readouts/metadata* (default True, small)
    with_readouts : all pre-computed readouts (TACs, SUVs per region)
    ct_reconstruction, pet_reconstruction : series used by the "static" preset
    dry_run : only list what would be downloaded (count, size), download nothing

    Returns the path of the split folder (<target_dir>/train), which can be
    used as cohort folder (one sub-folder per subject).
    """
    from huggingface_hub import HfApi, snapshot_download

    api = HfApi()
    target_dir = Path(target_dir)
    available, root_files = _split_root(api)
    if subjects is not None:
        unknown = sorted(set(subjects) - set(available))
        if unknown:
            raise ValueError(f"Unknown subjects {unknown}. Available ({len(available)}): "
                             f"{', '.join(available[:8])}, ...")
    elif n_subjects:
        subjects = available[:n_subjects]

    allow, ignore = build_patterns(subjects, content, with_totalsegmentator, with_mri,
                                   with_clinical, with_readouts, ct_reconstruction,
                                   pet_reconstruction,
                                   root_files=root_files)
    files, size = plan_download(allow, ignore, api)
    n_subs = len({f.path.split("/")[1] for f in files if f.path.split("/")[1].startswith("sub-")})
    print(f"Multimodal-HC: {len(files)} files, {n_subs} subjects, {size / 1e9:.1f} GB "
          f"(content '{content}')")
    if not files:
        raise RuntimeError("Nothing matches the selection.")

    if dry_run:
        for f in files[:15]:
            print(f"   {f.path}  ({(f.size or 0) / 1e6:.1f} MB)")
        if len(files) > 15:
            print(f"   ... {len(files) - 15} more")
        return target_dir / SPLIT

    target_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(target_dir).free
    if size > free:
        raise RuntimeError(f"Not enough disk space: need {size / 1e9:.1f} GB, "
                           f"{free / 1e9:.1f} GB free in {target_dir}")

    snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE, local_dir=str(target_dir),
                      allow_patterns=allow, ignore_patterns=ignore or None,
                      max_workers=max_workers)
    print(f"✅ Downloaded to {target_dir / SPLIT}\n"
          f"   Please cite: Hinge et al., Scientific Data (2026), doi:10.1038/s41597-026-08157-4")
    return target_dir / SPLIT


def main():
    p = argparse.ArgumentParser(description="Download the Multimodal-HC PET/CT dataset")
    p.add_argument("--target-dir", type=Path, required=True)
    p.add_argument("--subjects", nargs="+", default=None, help="e.g. sub-000 sub-001")
    p.add_argument("--n-subjects", type=int, default=None, help="first n subjects only")
    p.add_argument("--content", choices=CONTENT_PRESETS, default="static")
    p.add_argument("--with-totalsegmentator", action="store_true")
    p.add_argument("--with-mri", action="store_true")
    p.add_argument("--no-clinical", action="store_true", help="images only, no clinical values")
    p.add_argument("--with-readouts", action="store_true",
                   help="also all pre-computed readouts (TACs, SUVs per region)")
    p.add_argument("--dry-run", action="store_true", help="only show what would be downloaded")
    a = p.parse_args()
    download_multimodal_hc(a.target_dir, a.subjects, a.n_subjects, a.content,
                           a.with_totalsegmentator, a.with_mri, not a.no_clinical,
                           a.with_readouts,
                           dry_run=a.dry_run)


if __name__ == "__main__":
    main()