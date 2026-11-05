"""
Compare the segmentation tools (TotalSegmentator, MOOSE, CADS - COMPARISON_TOOLS
in config.py) pairwise: SUV_mean and SUV_max of the organs both tools of a pair
provide, without erosion and with 1 voxel eroded around every mask.

All tools are applied to the same PET of each patient, so differences come only
from the segmentation. Values are raw SUVs (not normalised); the IVC reference
(L2) is compared separately.

Settings in config.py. Output in <OUTPUT_DIR>/<cohort>_SUV_extraction/tool_comparison/:
    per_patient_values.csv          one row per patient/organ/erosion/tool
    reference_IVC_L2.csv            IVC reference per tool
    organ_availability.csv          which organs exist in which tool, per patient
    agreement_summary_all_pairs.csv statistics of every pair in one table
    plots/boxplots_<metric>.png     per organ: every tool x non-eroded/eroded
    TS_vs_MOOSE/, TS_vs_CADS/, MOOSE_vs_CADS/   one folder per pair:
        paired_values.csv           both tools side by side + differences
        agreement_summary.csv       statistics per organ x metric x erosion
        plots/                      scatter_<metric>, bland_altman_<metric>,
                                    relative_bias_overview
    erosion_effect/                 eroded vs non-eroded within each tool (all
                                    organs of a tool, incl. single-tool ones like fat):
        erosion_paired_values.csv, erosion_summary.csv,
        plots/ (erosion_scatter_<metric>, erosion_overview)

Run:  python compare_segmentations.py [--data-dir <cohort folder>] [--output-dir <folder>]
"""
import pandas as pd

import config as cfg
from Analysis.organ_definitions import organ_config
from Analysis.segmentation_comparison import (TOOL_LABEL, find_organ_availability, extract_patient,
                                              to_pairs, summarize, plot_comparison, tool_pairs, pair_name,
                                              erosion_pairs, summarize_erosion, plot_erosion, plot_boxplots,
                                              REFERENCE_LABEL)


def _console_pair_summary(summary, a, b):
    organ_rows = summary[summary.organ != REFERENCE_LABEL].copy()
    if organ_rows.empty:
        return
    organ_rows["condition"] = (organ_rows["metric"].str.replace("suv_", "SUV")
                               + organ_rows["erosion"].map(lambda e: " eroded" if e else " full"))
    for value, title in (("rel_bias_pct", f"Relative bias {TOOL_LABEL[b]} vs {TOOL_LABEL[a]} [% of pair mean]"),
                         ("icc_a1", f"ICC(A,1) {TOOL_LABEL[a]} / {TOOL_LABEL[b]}")):
        table = organ_rows.pivot_table(index="organ", columns="condition", values=value,
                                       aggfunc="first", sort=False)
        print(f"\n{title}:")
        print(table.round(2).to_string())
    ref = summary[summary.organ == REFERENCE_LABEL]
    if len(ref) and ref.iloc[0]["n"]:
        r = ref.iloc[0]
        print(f"{REFERENCE_LABEL}: n={r['n']}, bias {r.get('bias', float('nan')):.3f} SUV "
              f"({r.get('rel_bias_pct', float('nan')):.1f}%), ICC {r.get('icc_a1', float('nan')):.2f}")


def run_comparison(data_dir, output_dir, standalone=False, excluded_patients=None, tools=None):
    """
    Compare the segmentation tools of one cohort pairwise (every pair of
    COMPARISON_TOOLS); called by main.py (COMPARE_TOOLS) or directly.

    Output: <cohort>_SUV_extraction/tool_comparison/
        per_patient_values.csv, reference_IVC_L2.csv, organ_availability.csv,
        segmentation_status.csv, excluded_patients.csv,
        agreement_summary_all_pairs.csv
        plots/boxplots_<metric>.png        all tools, non-eroded vs eroded
        erosion_effect/                    eroded vs non-eroded within each tool
        TS_vs_MOOSE/, TS_vs_CADS/, MOOSE_vs_CADS/
            paired_values.csv, agreement_summary.csv,
            plots/ (scatter, Bland-Altman, relative bias)
    """
    tools = list(tools or cfg.COMPARISON_TOOLS)
    if len(tools) < 2:
        print("Comparison needs at least two tools (COMPARISON_TOOLS) - skipped.")
        return
    if standalone:     # main.py has already started the issue log of this run
        from Image_loading.issue_log import set_issue_log
        from Image_loading.Image_Loader import reset_pet_source_decisions
        cohort_dir, cohort = cfg.cohort_output_dir(output_dir, data_dir)
        set_issue_log(cohort_dir / f"{cohort}_patient_issues.txt", reset=True)
        reset_pet_source_decisions()
    print(f"\n#################### Comparison {' / '.join(TOOL_LABEL[t] for t in tools)} ####################")
    print(f"Cohort: {data_dir.resolve()}")
    out = cfg.cohort_output_dir(output_dir, data_dir)[0] / "tool_comparison"
    out.mkdir(parents=True, exist_ok=True)

    organs, combinations = {}, {}
    for tool in tools:
        organs[tool], combinations[tool] = organ_config(tool)

    # every compared tool needs its segmentations; complete missing ones if enabled
    from Image_loading.segmentation_manager import ensure_segmentations
    status = ensure_segmentations(
        data_dir, tools,
        moose_models=cfg.MOOSE_MODELS, totalseg_tasks=cfg.TOTALSEG_TASKS,
        segment_missing=cfg.SEGMENT_MISSING, force=cfg.FORCE_RESEGMENTATION,
        ct_name_hints=cfg.CT_NAME_HINTS, moose_accelerator=cfg.MOOSE_ACCELERATOR,
        totalseg_fast=cfg.TOTALSEG_FAST, cads_tasks=cfg.CADS_TASKS, cads_license=cfg.CADS_LICENSE,
        cads_use_cpu=cfg.CADS_USE_CPU, cads_command=cfg.CADS_COMMAND, cads_one_run=cfg.CADS_ONE_RUN_PER_PATIENT)
    status.to_csv(out / "segmentation_status.csv", index=False, sep=";")

    from Image_loading.folder_detection import list_patient_dirs
    patients = [p.name for p in list_patient_dirs(data_dir)]
    rows, references, availability, failed = [], [], [], []

    for i, patient in enumerate(patients, 1):
        print(f"\n========== [{i}/{len(patients)}] {patient}")
        try:
            available, avail = find_organ_availability(
                data_dir / patient, tools, organs, combinations,
                all_shared=cfg.COMPARE_ALL_SHARED_REGIONS,
                moose_model_priority=cfg.MOOSE_SUBMODEL_PRIORITY, cads_task_priority=cfg.CADS_TASKS)
            availability += [{"patient": patient, **a} for a in avail]
            # tools without any segmentation for this patient are left out (other pairs still use it)
            organs_by_tool = {t: available[t] for t in tools if available[t]}
            missing_tools = [TOOL_LABEL[t] for t in tools if t not in organs_by_tool]
            if missing_tools:
                print(f"⚠️ no segmentation from {', '.join(missing_tools)} - compared without")
            if len(organs_by_tool) < 2:
                print("⚠️ fewer than two tools with segmentations - patient skipped")
                failed.append(patient)
                continue
            r, ref = extract_patient(
                data_dir, patient, organs_by_tool, combinations, cfg.REFERENCE_ORGAN, cfg.REFERENCE_VERTEBRA,
                pet_name_hints=cfg.PET_NAME_HINTS, moose_model_priority=cfg.MOOSE_SUBMODEL_PRIORITY,
                erosion_voxels=cfg.COMPARISON_EROSION_VOXELS,
                suv_outlier_threshold=cfg.SUV_MEAN_OUTLIER_THRESHOLD,
                pet_nifti_hints=cfg.PET_NIFTI_HINTS, cads_task_priority=cfg.CADS_TASKS)
            rows += r
            references.append(ref)
        except Exception as e:
            print(f"❌ {patient}: {e}")
            failed.append(patient)

    if not rows:
        print("No results.")
        return

    long_df = pd.DataFrame(rows)
    reference_df = pd.DataFrame(references)

    # ---------------- patient filter ----------------
    excluded = set(excluded_patients or [])
    if cfg.OUTLIER_FILTER:
        from Analysis.outlier_filter import find_outlier_patients
        from Image_loading.issue_log import log_issue
        reports = []
        for tool in tools:     # own check on the non-eroded SUV_means of each tool
            sub = long_df[(long_df.tool == tool) & (long_df.erosion == 0)]
            if sub.empty:
                continue
            wide = (sub.pivot_table(index="patient", columns="organ", values="suv_mean", aggfunc="first")
                    .add_suffix("_mean_automatic").reset_index())
            rep = find_outlier_patients(wide, [c for c in wide if c != "patient"], patient_col="patient",
                                        n_sd=cfg.OUTLIER_FILTER_SD, min_organs=cfg.OUTLIER_FILTER_MIN_ORGANS,
                                        method=cfg.OUTLIER_FILTER_METHOD)
            reports.append(rep.assign(tool=tool))
        if reports:
            report = pd.concat(reports, ignore_index=True)
            for _, r in report[report["excluded"]].iterrows():
                if r["patient"] not in excluded:
                    log_issue(r["patient"], "filter", f"excluded from comparison ({r['tool']}): "
                                                      f"{r['n_outlier_organs']} organs > {cfg.OUTLIER_FILTER_SD:g} SD: "
                                                      f"{r['outlier_organs']}")
            excluded |= set(report.loc[report["excluded"], "patient"])
            report[report["n_outlier_organs"] > 0].to_csv(out / "excluded_patients.csv", index=False, sep=";")
    if excluded:
        print(f"\nExcluded from the comparison: {sorted(excluded)}")
        long_df = long_df[~long_df["patient"].isin(excluded)]
        reference_df = reference_df[~reference_df["patient"].isin(excluded)]

    csv = dict(index=False, sep=";", decimal=".")
    avail_df = pd.DataFrame(availability)
    long_df.to_csv(out / "per_patient_values.csv", **csv)
    reference_df.to_csv(out / "reference_IVC_L2.csv", **csv)
    avail_df.to_csv(out / "organ_availability.csv", **csv)
    plot_boxplots(long_df, out / "plots")

    # ---------------- every pair of tools ----------------
    all_summaries = []
    for a, b in tool_pairs(tools):
        pairs = to_pairs(long_df, (a, b))
        if pairs.empty:
            print(f"\n⚠️ {TOOL_LABEL[a]} vs {TOOL_LABEL[b]}: no organ both provide - skipped")
            continue
        pout = out / pair_name(a, b)
        pout.mkdir(exist_ok=True)
        summary = summarize(pairs, reference_df, (a, b))
        pairs.to_csv(pout / "paired_values.csv", **csv)
        summary.round(4).to_csv(pout / "agreement_summary.csv", **csv)
        plot_comparison(pairs, summary, pout / "plots", pair=(a, b))
        all_summaries.append(summary.assign(pair=pair_name(a, b)))
        print(f"\n==================== {TOOL_LABEL[a]} vs {TOOL_LABEL[b]} ====================")
        _console_pair_summary(summary, a, b)
    if all_summaries:
        combined = pd.concat(all_summaries, ignore_index=True)
        combined = combined[["pair"] + [c for c in combined.columns if c != "pair"]]
        combined.round(4).to_csv(out / "agreement_summary_all_pairs.csv", **csv)

    # ---------------- eroded vs non-eroded, per tool ----------------
    eout = out / "erosion_effect"
    eout.mkdir(exist_ok=True)
    epairs = erosion_pairs(long_df)
    esummary = summarize_erosion(epairs)
    epairs.to_csv(eout / "erosion_paired_values.csv", **csv)
    esummary.round(4).to_csv(eout / "erosion_summary.csv", **csv)
    plot_erosion(epairs, esummary, eout / "plots")

    view = esummary.copy()
    view["condition"] = (view["tool"].map(TOOL_LABEL).str[:5] + " "
                         + view["metric"].str.replace("suv_", "SUV"))
    table = view.pivot_table(index="organ", columns="condition", values="rel_bias_pct",
                             aggfunc="first", sort=False)
    removed = (view[view.metric == "suv_mean"]
               .pivot_table(index="organ", columns="tool", values="median_voxels_removed_pct",
                            aggfunc="first", sort=False)
               .rename(columns=lambda t: f"vox removed {TOOL_LABEL[t][:5]} %"))
    pd.set_option("display.width", 220)
    print("\nErosion effect: change of eroded vs non-eroded [% of non-eroded value]:")
    print(table.join(removed).round(1).to_string())

    status_cols = [c for c in avail_df.columns if c.endswith("_status")]
    never = [o for o, g in avail_df.groupby("organ")
             if ((g[status_cols] == "ok").sum(axis=1) < 2).all()]
    if never:
        print(f"\nNot compared (provided by fewer than two tools for every patient): {', '.join(never)}")
        print("  -> details in organ_availability.csv")
    if failed:
        print(f"\n⚠️ {len(failed)} patients not compared: {failed}")
    print(f"\n✅ Results in {out}")


def main():
    data_dir, output_dir, _ = cfg.parse_args("Compare the segmentation tools (TotalSegmentator, MOOSE, CADS)")
    run_comparison(data_dir, output_dir, standalone=True)


if __name__ == "__main__":
    main()