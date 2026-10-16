"""
Compare MOOSE and TotalSegmentator: SUV_mean and SUV_max of the organs both
tools provide, without erosion and with 1 voxel eroded around every mask.

Both segmentations are applied to the same PET of each patient, so differences
come only from the segmentation. Values are raw SUVs (not normalised); the IVC
reference (L2) of both tools is compared separately.

Settings in config.py. Output in <OUTPUT_DIR>/<cohort>_SUV_extraction/TS_vs_MOOSE/:
    per_patient_values.csv   one row per patient/organ/erosion/tool
    paired_values.csv        both tools side by side + differences
    agreement_summary.csv    statistics per organ x metric x erosion
    reference_IVC_L2.csv     IVC reference per tool
    organ_availability.csv   which organs exist in which tool, per patient
    plots/                   PNG: boxplots_<metric> (per organ: TotalSeg/MOOSE x
                             non-eroded/eroded), scatter_<metric>,
                             bland_altman_<metric>, relative_bias_overview
    erosion_effect/          eroded vs non-eroded within each tool (all organs
                             of a tool, incl. MOOSE-only ones like fat):
        erosion_paired_values.csv, erosion_summary.csv,
        plots/ (PNG: erosion_scatter_<metric>, erosion_overview)

Run:  python compare_segmentations.py [--data-dir <cohort folder>] [--output-dir <folder>]
"""
import pandas as pd

import config as cfg
from Analysis.organ_definitions import organ_config
from Analysis.segmentation_comparison import (TOOLS, TOOL_LABEL, find_shared_organs, extract_patient,
                                              to_pairs, summarize, plot_comparison,
                                              erosion_pairs, summarize_erosion, plot_erosion, plot_boxplots,
                                              REFERENCE_LABEL)


def run_comparison(data_dir, output_dir, standalone=False):
    """Compare both tools for one cohort; called by main.py (COMPARE_TOOLS) or directly."""
    if standalone:     # main.py has already started the issue log of this run
        from Image_loading.issue_log import set_issue_log
        from Image_loading.Image_Loader import reset_pet_source_decisions
        cohort_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
        set_issue_log(cohort_dir / f"{cohort}_patient_issues.txt", reset=True)
        reset_pet_source_decisions()
    print("\n#################### Comparison TotalSegmentator vs MOOSE ####################")
    print(f"Cohort: {data_dir.resolve()}")
    out = cfg.cohort_output_dir(output_dir, data_dir)[0] / "TS_vs_MOOSE"
    out.mkdir(parents=True, exist_ok=True)

    organs, combinations = {}, {}
    for tool in TOOLS:
        organs[tool], combinations[tool] = organ_config(tool)

    # both tools need their segmentations; complete missing ones if enabled
    from Image_loading.segmentation_manager import ensure_segmentations
    status = ensure_segmentations(
        data_dir, list(TOOLS),
        moose_models=cfg.MOOSE_MODELS, totalseg_tasks=cfg.TOTALSEG_TASKS,
        segment_missing=cfg.SEGMENT_MISSING, force=cfg.FORCE_RESEGMENTATION,
        ct_name_hints=cfg.CT_NAME_HINTS, moose_accelerator=cfg.MOOSE_ACCELERATOR,
        totalseg_fast=cfg.TOTALSEG_FAST)
    status.to_csv(out / "segmentation_status.csv", index=False, sep=";")

    patients = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    rows, references, availability, failed = [], [], [], []

    for i, patient in enumerate(patients, 1):
        print(f"\n========== [{i}/{len(patients)}] {patient}")
        try:
            shared, avail = find_shared_organs(
                data_dir / patient, organs, combinations,
                all_shared=cfg.COMPARE_ALL_SHARED_REGIONS,
                moose_model_priority=cfg.MOOSE_SUBMODEL_PRIORITY)
            availability += [{"patient": patient, **a} for a in avail]
            if not shared:
                print("⚠️ no organs in both tools")
                failed.append(patient)
                continue
            print(f"Shared organs ({len(shared)}): {', '.join(shared)}")
            # for eroded vs non-eroded: also the organs only one tool provides (complete)
            extra = {tool: [a["organ"] for a in avail
                            if a[f"{tool}_status"] == "ok" and a["organ"] not in shared]
                     for tool in TOOLS}

            r, ref = extract_patient(
                data_dir, patient, shared, combinations, cfg.REFERENCE_ORGAN, cfg.REFERENCE_VERTEBRA,
                pet_name_hints=cfg.PET_NAME_HINTS, moose_model_priority=cfg.MOOSE_SUBMODEL_PRIORITY,
                erosion_voxels=cfg.COMPARISON_EROSION_VOXELS, extra_organs=extra,
                suv_outlier_threshold=cfg.SUV_MEAN_OUTLIER_THRESHOLD,
                pet_nifti_hints=cfg.PET_NIFTI_HINTS)
            rows += r
            references.append(ref)
        except Exception as e:
            print(f"❌ {patient}: {e}")
            failed.append(patient)

    if not rows:
        print("No results.")
        return

    long_df = pd.DataFrame(rows)
    pairs = to_pairs(long_df)
    reference_df = pd.DataFrame(references)
    summary = summarize(pairs, reference_df)
    avail_df = pd.DataFrame(availability)

    csv = dict(index=False, sep=";", decimal=".")
    long_df.to_csv(out / "per_patient_values.csv", **csv)
    pairs.to_csv(out / "paired_values.csv", **csv)
    summary.round(4).to_csv(out / "agreement_summary.csv", **csv)
    reference_df.to_csv(out / "reference_IVC_L2.csv", **csv)
    avail_df.to_csv(out / "organ_availability.csv", **csv)
    plot_comparison(pairs, summary, out / "plots")
    plot_boxplots(long_df, out / "plots")

    # ---------------- eroded vs non-eroded, per tool ----------------
    eout = out / "erosion_effect"
    eout.mkdir(exist_ok=True)
    epairs = erosion_pairs(long_df)
    esummary = summarize_erosion(epairs)
    epairs.to_csv(eout / "erosion_paired_values.csv", **csv)
    esummary.round(4).to_csv(eout / "erosion_summary.csv", **csv)
    plot_erosion(epairs, esummary, eout / "plots")

    # ---------------- console overview ----------------
    not_shared = (avail_df.assign(shared=(avail_df.filter(like="_status") == "ok").all(axis=1))
                  .groupby("organ")["shared"].sum())
    never = [o for o, n in not_shared.items() if n == 0]
    if never:
        print(f"\nNot compared (missing/partial in a tool for all patients): {', '.join(never)}")
        print("  -> details in organ_availability.csv")

    pd.set_option("display.width", 200)
    organ_rows = summary[summary.organ != REFERENCE_LABEL].copy()
    organ_rows["condition"] = (organ_rows["metric"].str.replace("suv_", "SUV")
                               + organ_rows["erosion"].map(lambda e: " eroded" if e else " full"))
    for value, title in (("rel_bias_pct", "Relative bias MOOSE vs TotalSegmentator [% of pair mean]"),
                         ("icc_a1", "ICC(A,1)")):
        table = organ_rows.pivot_table(index="organ", columns="condition", values=value,
                                       aggfunc="first", sort=False)
        print(f"\n{title}:")
        print(table.round(2).to_string())
    ref = summary[summary.organ == REFERENCE_LABEL]
    if len(ref):
        r = ref.iloc[0]
        print(f"\n{REFERENCE_LABEL}: n={r['n']}, bias {r.get('bias', float('nan')):.3f} SUV "
              f"({r.get('rel_bias_pct', float('nan')):.1f}%), ICC {r.get('icc_a1', float('nan')):.2f}")
    view = esummary.copy()
    view["condition"] = (view["tool"].map(TOOL_LABEL).str[:5] + " "
                         + view["metric"].str.replace("suv_", "SUV"))
    table = view.pivot_table(index="organ", columns="condition", values="rel_bias_pct",
                             aggfunc="first", sort=False)
    removed = (view[view.metric == "suv_mean"]
               .pivot_table(index="organ", columns="tool", values="median_voxels_removed_pct",
                            aggfunc="first", sort=False)
               .rename(columns=lambda t: f"vox removed {TOOL_LABEL[t][:5]} %"))
    print("\nErosion effect: change of eroded vs non-eroded [% of non-eroded value]:")
    print(table.join(removed).round(1).to_string())

    if failed:
        print(f"\n⚠️ {len(failed)} patients not compared: {failed}")
    print(f"\n✅ Results in {out}")


def main():
    data_dir, output_dir, _ = cfg.parse_args("Compare MOOSE and TotalSegmentator SUVs")
    run_comparison(data_dir, output_dir, standalone=True)


if __name__ == "__main__":
    main()