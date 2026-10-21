"""
Agreement between segmentation tools (TotalSegmentator, MOOSE, CADS) in SUVs,
pairwise for every pair of tools, plus the effect of erosion within each tool.

Per patient both segmentations are applied to the SAME PET (same SUV
conversion), for the organs both tools provide, once without erosion and once
with 1 voxel removed around every mask (3x3x3 structuring element, as in
erode_organ_masks). Compared are SUV_mean and SUV_max.

An organ counts as shared only if it is an exact region in both tools, or if
ALL parts of its combination exist (e.g. MOOSE heart = myocardium + 4
chambers). Substring matches are never used here, because they could compare
different structures (heart vs heart_atrium_left).

Masks too small to erode (erode_organ_masks keeps them uneroded, so eroded
and non-eroded values would be identical) get NaN as eroded value
(erosion_fallback = True): they are excluded from every eroded comparison,
their non-eroded value is kept.

Statistics per tool pair, organ, metric and erosion level (second - first tool):
    n, means, bias (mean difference), SD of differences, Bland-Altman 95%
    limits of agreement, relative bias in % of the pair mean, ICC(A,1)
    (two-way, absolute agreement, single measures), Pearson r, Spearman rho,
    Wilcoxon signed-rank p.
"""

import numpy as np
import pandas as pd

TOOLS = ("totalsegmentator", "moose", "cads")
TOOL_LABEL = {"totalsegmentator": "TotalSegmentator", "moose": "MOOSE", "cads": "CADS"}
TOOL_SHORT = {"totalsegmentator": "ts", "moose": "moose", "cads": "cads"}      # column names
TOOL_TAG = {"totalsegmentator": "TS", "moose": "MOOSE", "cads": "CADS"}        # folder names
TOOL_COLOR = {"totalsegmentator": "tab:green", "moose": "tab:purple", "cads": "tab:orange"}


def pair_name(a, b):
    """('totalsegmentator', 'moose') -> 'TS_vs_MOOSE'"""
    return f"{TOOL_TAG[a]}_vs_{TOOL_TAG[b]}"


def tool_pairs(tools):
    """All pairs in the given order: TS-MOOSE, TS-CADS, MOOSE-CADS."""
    tools = list(tools)
    return [(a, b) for i, a in enumerate(tools) for b in tools[i + 1:]]
REFERENCE_LABEL = "IVC reference (L2)"


# ---------------------------------------------------------------------------
# Shared organs
# ---------------------------------------------------------------------------
def _status(organ, names_lower, combination):
    """'ok', 'partial' (only some parts of a combination) or 'missing'."""
    if organ in combination:
        present = [c.lower() in names_lower for c in combination[organ]]
        if all(present):
            return "ok"
        return "partial" if any(present) else "missing"
    return "ok" if organ.lower() in names_lower else "missing"


def find_organ_availability(patient_path, tools, organs, combinations, all_shared=False,
                            moose_model_priority=None, cads_task_priority=None):
    """
    Which organs each tool provides for one patient.

    Parameters
    ----------
    tools : tools to check
    organs : dict tool -> list of organ names (candidates)
    combinations : dict tool -> combination_logic
    all_shared : also every region that exists under the same name in at
        least two tools (ribs, vertebrae, muscles, ...).

    Returns
    -------
    available : dict tool -> organs that tool provides completely
    availability : list of dicts (organ, <tool>_status) for reporting;
        status 'ok', 'partial' (incomplete combination) or 'missing'
    """
    from Image_loading.segmentation_sources import get_segmentation_source

    names = {}
    for tool in tools:
        src = get_segmentation_source(patient_path, tool, None, verbose=False,
                                      moose_model_priority=moose_model_priority,
                                      cads_task_priority=cads_task_priority)
        names[tool] = {n.lower(): n for n in (src.names() if src else [])}

    candidates = list(dict.fromkeys(
        [o for t in tools for o in organs[t]] + [o for t in tools for o in combinations[t]]))
    if all_shared:
        counts = {}
        for t in tools:
            for n in names[t]:
                counts[n] = counts.get(n, 0) + 1
        display = {n: v for t in tools for n, v in names[t].items()}
        candidates += [display[n] for n in sorted(counts) if counts[n] >= 2
                       and display[n] not in candidates]

    available, availability = {t: [] for t in tools}, []
    for organ in candidates:
        st = {tool: _status(organ, names[tool], combinations[tool]) for tool in tools}
        availability.append({"organ": organ, **{f"{t}_status": s for t, s in st.items()}})
        for t, status in st.items():
            if status == "ok":
                available[t].append(organ)
    return available, availability


def find_shared_organs(patient_path, organs, combinations, all_shared=False,
                       moose_model_priority=None, tools=("totalsegmentator", "moose")):
    """Organs available in all given tools (kept for compatibility)."""
    available, availability = find_organ_availability(patient_path, tools, organs, combinations,
                                                       all_shared, moose_model_priority)
    shared = [o for o in available[tools[0]] if all(o in available[t] for t in tools)]
    return shared, availability


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _organ_values(arr):
    vals = arr[arr > 0]
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(vals.mean()), float(vals.max()), int(vals.size)


def extract_patient(data_dir, patient, organs_by_tool, combinations, reference_organ,
                    reference_vertebra, pet_name_hints=None, moose_model_priority=None,
                    erosion_voxels=1, suv_outlier_threshold=20.0, pet_nifti_hints=None,
                    cads_task_priority=None):
    """
    SUV_mean / SUV_max per tool, organ and erosion level for one patient.

    organs_by_tool : dict tool -> organs that tool provides (find_organ_availability);
                     the tools compared are the keys.

    Returns (rows, reference_row). rows: list of dicts with patient, organ,
    erosion, tool, suv_mean, suv_max, voxels, erosion_fallback.
    """
    from Image_loading.Image_Loader import PET_Organ_Cropper, erode_organ_masks

    rows, reference = [], {"patient": patient}
    tools = list(organs_by_tool)

    def _load(tool):
        result = PET_Organ_Cropper(
            data_dir=data_dir, patient=patient, organs_of_interest=organs_by_tool[tool],
            combination_logic=combinations[tool], norm_reference=reference_organ,
            reference_vertebra=reference_vertebra, segmentation_backend=tool,
            pet_name_hints=pet_name_hints, moose_model_priority=moose_model_priority,
            suv_outlier_threshold=suv_outlier_threshold, pet_nifti_hints=pet_nifti_hints,
            cads_task_priority=cads_task_priority)
        if result is None:
            raise RuntimeError(f"{TOOL_LABEL[tool]} extraction failed")
        return result

    results = {tool: _load(tool) for tool in tools}
    # same PET for both tools: if the second switched to the PET NIfTI, redo the first
    if len({r["pet_source"] for r in results.values()}) > 1:
        for tool in tools:
            if results[tool]["pet_source"] == "dicom":
                print(f"↻ {patient}: re-extracting {TOOL_LABEL[tool]} with the PET NIfTI")
                results[tool] = _load(tool)
    reference["pet_source"] = results[tools[0]]["pet_source"]
    reference["pet_issue"] = results[tools[0]]["pet_issue"]

    for tool in tools:
        result, tool_organs = results[tool], organs_by_tool[tool]

        reference[f"{tool}_suv_mean"] = result["reference_SUV_mean"]
        reference[f"{tool}_method"] = result["reference_method"]
        reference[f"{tool}_voxels"] = result["reference_voxels"]

        # the reference organ is returned restricted to the vertebra level
        organs = {o: a for o, a in result["organs"].items()
                  if o in tool_organs and o != reference_organ}
        eroded = erode_organ_masks(organs, erosion_config={"DEFAULT": erosion_voxels})

        for organ in tool_organs:
            if organ == reference_organ:
                continue
            for erosion, data in ((0, organs), (erosion_voxels, eroded)):
                if organ in data:
                    mean, mx, vox = _organ_values(data[organ])
                else:
                    mean, mx, vox = np.nan, np.nan, 0
                fallback = False
                if erosion > 0 and organ in organs:
                    # erode_organ_masks keeps a mask uneroded if too little would
                    # remain. An erosion always removes voxels, so an unchanged
                    # voxel count (= identical values) means it was not performed:
                    # the "eroded" value is removed from every comparison.
                    fallback = vox == _organ_values(organs[organ])[2] and vox > 0
                    if fallback:
                        mean, mx, vox = np.nan, np.nan, np.nan
                rows.append({"patient": patient, "organ": organ, "erosion": erosion,
                             "tool": tool, "suv_mean": mean, "suv_max": mx,
                             "voxels": vox, "erosion_fallback": fallback})
    return rows, reference


def to_pairs(long_df, pair=("totalsegmentator", "moose")):
    """
    Long table -> one row per patient/organ/erosion with the two tools of
    `pair` side by side, for the organs both provide for that patient.
    """
    a, b = pair
    long_df = long_df[long_df["tool"].isin(pair)]
    if long_df.empty:
        return pd.DataFrame()
    n_tools = long_df.groupby(["patient", "organ"])["tool"].transform("nunique")
    long_df = long_df[n_tools == 2]
    if long_df.empty:              # no organ both tools provide
        return pd.DataFrame()
    wide = long_df.pivot_table(index=["patient", "organ", "erosion"], columns="tool",
                               values=["suv_mean", "suv_max", "voxels", "erosion_fallback"],
                               aggfunc="first")
    wide.columns = [f"{tool}_{val}" for val, tool in wide.columns]
    wide = wide.reset_index()
    for col in (f"{a}_erosion_fallback", f"{b}_erosion_fallback"):
        if col not in wide:
            wide[col] = False
    for metric in ("suv_mean", "suv_max"):
        t, m = wide[f"{a}_{metric}"], wide[f"{b}_{metric}"]
        wide[f"diff_{metric}"] = m - t
        wide[f"rel_diff_{metric}_pct"] = 100 * (m - t) / ((m + t) / 2)
    wide[f"voxel_ratio_{TOOL_SHORT[b]}_{TOOL_SHORT[a]}"] = wide[f"{b}_voxels"] / wide[f"{a}_voxels"]
    fb = wide[[f"{a}_erosion_fallback", f"{b}_erosion_fallback"]].fillna(False).astype(bool)
    wide["erosion_fallback"] = fb.any(axis=1)
    return wide


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def icc_a1(x, y):
    """ICC(A,1): two-way model, absolute agreement, single measures (McGraw & Wong)."""
    data = np.column_stack([x, y]).astype(float)
    n, k = data.shape
    if n < 3:
        return np.nan
    grand = data.mean()
    ssr = k * ((data.mean(axis=1) - grand) ** 2).sum()
    ssc = n * ((data.mean(axis=0) - grand) ** 2).sum()
    sse = ((data - grand) ** 2).sum() - ssr - ssc
    msr, msc, mse = ssr / (n - 1), ssc / (k - 1), sse / ((n - 1) * (k - 1))
    denom = msr + (k - 1) * mse + k * (msc - mse) / n
    return float((msr - mse) / denom) if denom > 0 else np.nan


def paired_stats(ts, mo, labels=("ts", "moose"), relative_to="pair_mean"):
    """
    Agreement statistics for paired arrays, differences = second - first
    (default: MOOSE - TotalSegmentator).

    labels      : names for the per-method mean/SD columns
    relative_to : 'pair_mean' (symmetric, for two methods) or 'first'
                  (change relative to the first, e.g. eroded vs non-eroded)
    """
    from scipy import stats

    ts, mo = np.asarray(ts, float), np.asarray(mo, float)
    ok = np.isfinite(ts) & np.isfinite(mo)
    ts, mo = ts[ok], mo[ok]
    n = len(ts)
    out = {"n": n}
    if n == 0:
        return out
    d = mo - ts
    rel = 100 * d / (ts if relative_to == "first" else (mo + ts) / 2)
    sd = d.std(ddof=1) if n > 1 else np.nan
    rel_sd = rel.std(ddof=1) if n > 1 else np.nan
    out.update({
        f"{labels[0]}_mean": ts.mean(), f"{labels[0]}_sd": ts.std(ddof=1) if n > 1 else np.nan,
        f"{labels[1]}_mean": mo.mean(), f"{labels[1]}_sd": mo.std(ddof=1) if n > 1 else np.nan,
        "bias": d.mean(), "sd_diff": sd,
        "loa_low": d.mean() - 1.96 * sd, "loa_high": d.mean() + 1.96 * sd,
        "rel_bias_pct": rel.mean(),
        "rel_loa_low_pct": rel.mean() - 1.96 * rel_sd, "rel_loa_high_pct": rel.mean() + 1.96 * rel_sd,
        "mean_abs_rel_diff_pct": np.abs(rel).mean(),
        "icc_a1": icc_a1(ts, mo),
    })
    if n >= 3 and ts.std() > 0 and mo.std() > 0:
        out["pearson_r"] = stats.pearsonr(ts, mo)[0]
        out["spearman_rho"] = stats.spearmanr(ts, mo)[0]
    if n >= 5 and np.any(d != 0):
        out["wilcoxon_p"] = stats.wilcoxon(mo, ts).pvalue
    return out


def summarize(pairs, reference_df=None, pair=("totalsegmentator", "moose")):
    """Summary table for one tool pair: one row per organ x metric x erosion
    (+ the IVC reference). Differences = second tool - first tool."""
    a, b = pair
    labels = (TOOL_SHORT[a], TOOL_SHORT[b])
    ratio = f"voxel_ratio_{TOOL_SHORT[b]}_{TOOL_SHORT[a]}"
    rows = []
    for (organ, erosion), grp in pairs.groupby(["organ", "erosion"], sort=False):
        n_fallback = 0
        if erosion > 0:
            n_fallback = int(grp["erosion_fallback"].sum())
            grp = grp[~grp["erosion_fallback"]]      # only truly eroded pairs
        for metric in ("suv_mean", "suv_max"):
            st = paired_stats(grp[f"{a}_{metric}"], grp[f"{b}_{metric}"], labels=labels)
            vr = grp[ratio].replace([np.inf, -np.inf], np.nan)
            rows.append({"organ": organ, "metric": metric, "erosion": erosion,
                         "excluded_erosion_fallback": n_fallback,
                         f"median_{ratio}": vr.median(), **st})
    if reference_df is not None and len(reference_df) and f"{a}_suv_mean" in reference_df \
            and f"{b}_suv_mean" in reference_df:
        st = paired_stats(reference_df[f"{a}_suv_mean"], reference_df[f"{b}_suv_mean"], labels=labels)
        rows.append({"organ": REFERENCE_LABEL, "metric": "suv_mean", "erosion": "in-plane",
                     "excluded_erosion_fallback": 0, **st})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots (PNG)
# ---------------------------------------------------------------------------
class _PngWriter:
    """Writes each figure as <out_dir>/<name>.png (150 dpi)."""

    def __init__(self, out_dir, dpi=150):
        from pathlib import Path
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.files = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def savefig(self, fig, name):
        path = self.out_dir / f"{name}.png"
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        self.files.append(path)


def _part(start, organs_per_page, n_organs):
    """'' if all organs fit in one figure, else '_part2' etc."""
    return "" if n_organs <= organs_per_page else f"_part{start // organs_per_page + 1}"


def plot_comparison(pairs, summary, out_dir, organs_per_page=16, pair=("totalsegmentator", "moose")):
    """PNGs in out_dir: scatter_<metric>, bland_altman_<metric>, relative_bias_overview."""
    a, b = pair
    la, lb = TOOL_LABEL[a], TOOL_LABEL[b]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    organs = list(dict.fromkeys(pairs["organ"]))
    erosions = sorted(pairs["erosion"].unique())
    style = {0: dict(marker="o", facecolors="none", edgecolors="tab:blue", label="no erosion"),
             1: dict(marker="o", color="tab:orange", label="eroded 1 voxel")}
    metric_label = {"suv_mean": "SUV$_{mean}$", "suv_max": "SUV$_{max}$"}

    def _grid(n):
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows), squeeze=False)
        for ax in axes.ravel()[n:]:
            ax.set_visible(False)
        return fig, axes.ravel()

    def _clean(grp):
        return grp[~grp["erosion_fallback"]] if grp["erosion"].iloc[0] > 0 else grp

    with _PngWriter(out_dir) as png:
        for metric in ("suv_mean", "suv_max"):
            for start in range(0, len(organs), organs_per_page):
                page = organs[start:start + organs_per_page]

                # --- scatter
                fig, axes = _grid(len(page))
                for ax, organ in zip(axes, page):
                    lo, hi = np.inf, -np.inf
                    for e in erosions:
                        g = pairs[(pairs.organ == organ) & (pairs.erosion == e)]
                        if g.empty:
                            continue
                        g = _clean(g)
                        x, y = g[f"{a}_{metric}"], g[f"{b}_{metric}"]
                        ax.scatter(x, y, s=22, **style.get(e, {}))
                        if len(x):
                            lo, hi = min(lo, x.min(), y.min()), max(hi, x.max(), y.max())
                    if np.isfinite(lo):
                        pad = 0.05 * (hi - lo or 1)
                        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
                    s = summary[(summary.organ == organ) & (summary.metric == metric)]
                    txt = "   ".join(f"ICC e{r['erosion']}: {r['icc_a1']:.2f}" for _, r in s.iterrows()
                                     if pd.notna(r.get("icc_a1", np.nan)))
                    ax.set_title(f"{organ}\n{txt}", fontsize=9)
                    ax.set_xlabel(f"{la} {metric_label[metric]}", fontsize=8)
                    ax.set_ylabel(f"{lb} {metric_label[metric]}", fontsize=8)
                    ax.tick_params(labelsize=7)
                axes[0].legend(fontsize=7, loc="upper left")
                fig.suptitle(f"{metric_label[metric]}: {lb} vs {la} (dashed = identity)")
                fig.tight_layout()
                png.savefig(fig, f"scatter_{metric}{_part(start, organs_per_page, len(organs))}"); plt.close(fig)

                # --- Bland-Altman
                fig, axes = _grid(len(page))
                for ax, organ in zip(axes, page):
                    for e in erosions:
                        g = pairs[(pairs.organ == organ) & (pairs.erosion == e)]
                        if g.empty:
                            continue
                        g = _clean(g)
                        t, m = g[f"{a}_{metric}"], g[f"{b}_{metric}"]
                        st = style.get(e, {})
                        color = st.get("color", st.get("edgecolors"))
                        ax.scatter((t + m) / 2, m - t, s=22, **st)
                        s = summary[(summary.organ == organ) & (summary.metric == metric)
                                    & (summary.erosion == e)]
                        if len(s) and pd.notna(s["bias"].iloc[0]):
                            ax.axhline(s["bias"].iloc[0], color=color, lw=1)
                            for lim in ("loa_low", "loa_high"):
                                if pd.notna(s[lim].iloc[0]):
                                    ax.axhline(s[lim].iloc[0], color=color, lw=0.8, ls=":")
                    ax.axhline(0, color="k", lw=0.6)
                    lo, hi = ax.get_ylim()
                    if hi - lo < 1e-9 or max(abs(lo), abs(hi)) < 1e-9:   # identical values
                        ax.set_ylim(-0.05, 0.05)
                    ax.set_title(organ, fontsize=9)
                    ax.set_xlabel(f"mean of both {metric_label[metric]}", fontsize=8)
                    ax.set_ylabel(f"{lb} − {la}", fontsize=8)
                    ax.tick_params(labelsize=7)
                axes[0].legend(fontsize=7, loc="upper left")
                fig.suptitle(f"Bland-Altman {metric_label[metric]} (solid = bias, dotted = 95% limits of agreement)")
                fig.tight_layout()
                png.savefig(fig, f"bland_altman_{metric}{_part(start, organs_per_page, len(organs))}"); plt.close(fig)

        # --- overview: relative bias with limits of agreement
        s = summary[summary.organ != REFERENCE_LABEL].copy()
        if len(s):
            fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.45 * len(organs) + 1.5)), sharey=True)
            y = np.arange(len(organs))
            for ax, metric in zip(axes, ("suv_mean", "suv_max")):
                for j, e in enumerate(erosions):
                    sub = s[(s.metric == metric) & (s.erosion == e)].set_index("organ").reindex(organs)
                    yy = y + (j - 0.5 * (len(erosions) - 1)) * 0.3
                    st = style.get(e, {})
                    color = st.get("color", st.get("edgecolors"))
                    err = np.vstack([sub["rel_bias_pct"] - sub["rel_loa_low_pct"],
                                     sub["rel_loa_high_pct"] - sub["rel_bias_pct"]])
                    ax.errorbar(sub["rel_bias_pct"], yy, xerr=err, fmt="o", color=color,
                                ms=5, capsize=2, label=st.get("label"))
                ax.axvline(0, color="k", lw=0.6)
                ax.set_title(f"{metric_label[metric]}: relative bias {lb} vs {la} (95% LoA)")
                ax.set_xlabel("% of pair mean")
                ax.grid(axis="x", alpha=0.3)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(organs)
            axes[0].invert_yaxis()
            axes[0].legend()
            fig.tight_layout()
            png.savefig(fig, "relative_bias_overview"); plt.close(fig)


# ---------------------------------------------------------------------------
# Eroded vs non-eroded (within each tool)
# ---------------------------------------------------------------------------
def erosion_pairs(long_df):
    """
    One row per tool/patient/organ: non-eroded vs eroded SUV_mean/max, voxels
    and the change caused by the erosion (eroded - non-eroded).
    Pairs where the mask was kept uneroded (too small) are flagged.
    """
    df = long_df.copy()
    df["level"] = np.where(df["erosion"] == 0, "full", "eroded")
    wide = df.pivot_table(index=["tool", "patient", "organ"], columns="level",
                          values=["suv_mean", "suv_max", "voxels", "erosion_fallback"],
                          aggfunc="first")
    wide.columns = [f"{lvl}_{val}" for val, lvl in wide.columns]
    wide = wide.reset_index()
    wide["erosion_fallback"] = wide.get("eroded_erosion_fallback", False)
    wide["erosion_fallback"] = wide["erosion_fallback"].fillna(False).astype(bool)
    wide = wide.drop(columns=[c for c in wide if c.endswith("_erosion_fallback")])
    for metric in ("suv_mean", "suv_max"):
        f, e = wide[f"full_{metric}"], wide[f"eroded_{metric}"]
        wide[f"change_{metric}"] = e - f
        wide[f"rel_change_{metric}_pct"] = 100 * (e - f) / f
    wide["voxels_removed_pct"] = 100 * (1 - wide["eroded_voxels"] / wide["full_voxels"])
    return wide


def summarize_erosion(epairs):
    """
    Statistics of the erosion effect per tool x organ x metric:
    change = eroded - non-eroded, relative change in % of the non-eroded value.
    Pairs kept uneroded are excluded (they would show no change).
    """
    rows = []
    for (tool, organ), grp in epairs.groupby(["tool", "organ"], sort=False):
        n_fallback = int(grp["erosion_fallback"].sum())
        grp = grp[~grp["erosion_fallback"]]
        removed = grp["voxels_removed_pct"].replace([np.inf, -np.inf], np.nan)
        for metric in ("suv_mean", "suv_max"):
            st = paired_stats(grp[f"full_{metric}"], grp[f"eroded_{metric}"],
                              labels=("full", "eroded"), relative_to="first")
            rows.append({"tool": tool, "organ": organ, "metric": metric,
                         "excluded_kept_uneroded": n_fallback,
                         "median_voxels_removed_pct": removed.median(), **st})
    return pd.DataFrame(rows)


def plot_erosion(epairs, esummary, out_dir, organs_per_page=16):
    """PNGs in out_dir: erosion_scatter_<metric> (both tools), erosion_overview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = TOOL_COLOR
    metric_label = {"suv_mean": "SUV$_{mean}$", "suv_max": "SUV$_{max}$"}
    organs = list(dict.fromkeys(epairs["organ"]))
    tools = [t for t in TOOLS if t in set(epairs["tool"])]
    data = epairs[~epairs["erosion_fallback"]]

    with _PngWriter(out_dir) as png:
        for metric in ("suv_mean", "suv_max"):
            for start in range(0, len(organs), organs_per_page):
                page = organs[start:start + organs_per_page]
                cols = min(4, len(page)); rows_n = int(np.ceil(len(page) / cols))
                fig, axes = plt.subplots(rows_n, cols, figsize=(4 * cols, 3.6 * rows_n), squeeze=False)
                axes = axes.ravel()
                for ax in axes[len(page):]:
                    ax.set_visible(False)
                for ax, organ in zip(axes, page):
                    lo, hi = np.inf, -np.inf
                    lines = []
                    for tool in tools:
                        g = data[(data.organ == organ) & (data.tool == tool)]
                        x, y = g[f"full_{metric}"], g[f"eroded_{metric}"]
                        ok = np.isfinite(x) & np.isfinite(y)
                        if not ok.any():
                            continue
                        ax.scatter(x[ok], y[ok], s=22, color=colors[tool], alpha=0.8,
                                   label=TOOL_LABEL[tool])
                        lo, hi = min(lo, x[ok].min(), y[ok].min()), max(hi, x[ok].max(), y[ok].max())
                        s = esummary[(esummary.tool == tool) & (esummary.organ == organ)
                                     & (esummary.metric == metric)]
                        if len(s) and pd.notna(s["rel_bias_pct"].iloc[0]):
                            lines.append(f"{TOOL_LABEL[tool][:5]} {s['rel_bias_pct'].iloc[0]:+.1f}%")
                    if np.isfinite(lo):
                        pad = 0.05 * (hi - lo or 1)
                        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8)
                    ax.set_title(f"{organ}\n{'   '.join(lines)}", fontsize=9)
                    ax.set_xlabel(f"non-eroded {metric_label[metric]}", fontsize=8)
                    ax.set_ylabel(f"eroded {metric_label[metric]}", fontsize=8)
                    ax.tick_params(labelsize=7)
                axes[0].legend(fontsize=7, loc="upper left")
                fig.suptitle(f"{metric_label[metric]}: eroded (1 voxel) vs non-eroded "
                             f"(dashed = identity, mean relative change per tool)")
                fig.tight_layout()
                png.savefig(fig, f"erosion_scatter_{metric}{_part(start, organs_per_page, len(organs))}"); plt.close(fig)

        # overview: relative change with 95% limits, both tools, plus voxels removed
        fig, axes = plt.subplots(1, 3, figsize=(18, max(4, 0.45 * len(organs) + 1.5)), sharey=True,
                                 gridspec_kw={"width_ratios": [1, 1, 0.6]})
        y = np.arange(len(organs))
        for ax, metric in zip(axes[:2], ("suv_mean", "suv_max")):
            width = 0.8 / max(len(tools), 1)
            for j, tool in enumerate(tools):
                sub = (esummary[(esummary.tool == tool) & (esummary.metric == metric)]
                       .set_index("organ").reindex(organs))
                if "rel_bias_pct" not in sub:
                    continue
                yy = y + (j - 0.5 * (len(tools) - 1)) * width
                err = np.vstack([sub["rel_bias_pct"] - sub["rel_loa_low_pct"],
                                 sub["rel_loa_high_pct"] - sub["rel_bias_pct"]])
                ax.errorbar(sub["rel_bias_pct"], yy, xerr=err, fmt="o", color=colors[tool],
                            ms=5, capsize=2, label=TOOL_LABEL[tool])
            ax.axvline(0, color="k", lw=0.6)
            ax.set_title(f"{metric_label[metric]}: change by erosion (95% limits)")
            ax.set_xlabel("% of non-eroded value")
            ax.grid(axis="x", alpha=0.3)
        width = 0.8 / max(len(tools), 1)
        for j, tool in enumerate(tools):
            sub = (esummary[(esummary.tool == tool) & (esummary.metric == "suv_mean")]
                   .set_index("organ").reindex(organs))
            axes[2].barh(y + (j - 0.5 * (len(tools) - 1)) * width, sub["median_voxels_removed_pct"],
                         height=width,
                         color=colors[tool], label=TOOL_LABEL[tool])
        axes[2].set_title("voxels removed (median)")
        axes[2].set_xlabel("%")
        axes[2].set_xlim(0, 100)
        axes[0].set_yticks(y)
        axes[0].set_yticklabels(organs)
        axes[0].invert_yaxis()
        axes[0].legend()
        fig.tight_layout()
        png.savefig(fig, "erosion_overview"); plt.close(fig)


def plot_boxplots(long_df, out_dir, organs_per_page=16):
    """
    PNG per metric: one panel per organ with two boxes per tool next to each
    other (non-eroded / eroded), e.g. TotalSegmentator, MOOSE, CADS, plus the
    individual patients as points. Organs that only some tools provide show
    only those tools' boxes. Not performed erosions (NaN) are not drawn.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = TOOL_COLOR
    short = {"totalsegmentator": "TotalSeg", "moose": "MOOSE", "cads": "CADS"}
    metric_label = {"suv_mean": "SUV$_{mean}$", "suv_max": "SUV$_{max}$"}
    tools = [t for t in TOOLS if t in set(long_df["tool"])]
    boxes = [(tool, erosion, 1.0 + 2.5 * i + erosion) for i, tool in enumerate(tools)
             for erosion in (0, 1)]

    df = long_df.copy()
    df["erosion"] = (df["erosion"] > 0).astype(int)
    # organs of all tools first, then organs only some tools provide
    tools_per_organ = df.dropna(subset=["suv_mean"]).groupby("organ")["tool"].unique()
    # only organs with at least one value (masks can be empty, e.g. outside the field of view)
    organs = [o for o in dict.fromkeys(df["organ"]) if len(tools_per_organ.get(o, []))]
    organs = sorted(organs, key=lambda o: -len(tools_per_organ.get(o, [])))
    if not organs:
        return
    rng = np.random.default_rng(0)

    with _PngWriter(out_dir) as png:
        for metric in ("suv_mean", "suv_max"):
            for start in range(0, len(organs), organs_per_page):
                page = organs[start:start + organs_per_page]
                cols = min(4, len(page)); rows_n = int(np.ceil(len(page) / cols))
                fig, axes = plt.subplots(rows_n, cols, figsize=((1.4 * len(tools) + 1.4) * cols, 3.8 * rows_n),
                                         squeeze=False)
                axes = axes.ravel()
                for ax in axes[len(page):]:
                    ax.set_visible(False)
                for ax, organ in zip(axes, page):
                    ticks, labels = [], []
                    for tool, erosion, pos in boxes:
                        vals = df[(df.organ == organ) & (df.tool == tool) & (df.erosion == erosion)][metric]
                        vals = vals[np.isfinite(vals)]
                        ticks.append(pos)
                        labels.append(f"{short[tool]}\n{'eroded' if erosion else 'full'}\nn={len(vals)}")
                        if vals.empty:
                            continue
                        bp = ax.boxplot(vals, positions=[pos], widths=0.7, patch_artist=True,
                                        showfliers=False, medianprops=dict(color="black"))
                        box = bp["boxes"][0]
                        box.set_facecolor(colors[tool])
                        box.set_alpha(0.35 if erosion else 0.75)
                        if erosion:
                            box.set_hatch("//")
                        ax.scatter(pos + rng.uniform(-0.18, 0.18, len(vals)), vals, s=10,
                                   color=colors[tool], edgecolor="black", linewidth=0.3, zorder=3)
                    ax.set_xticks(ticks)
                    ax.set_xticklabels(labels, fontsize=7)
                    ax.set_xlim(0.3, boxes[-1][2] + 0.7)
                    present = tools_per_organ.get(organ, [])
                    ax.set_title(organ if len(present) == len(tools) else
                                 f"{organ} ({', '.join(short[t] for t in tools if t in present)} only)",
                                 fontsize=9)
                    ax.set_ylabel(metric_label[metric], fontsize=8)
                    ax.tick_params(axis="y", labelsize=7)
                    ax.grid(axis="y", alpha=0.3)
                fig.suptitle(f"{metric_label[metric]} per organ: {' vs '.join(TOOL_LABEL[t] for t in tools)}, "
                             f"non-eroded vs eroded (1 voxel)")
                fig.tight_layout()
                png.savefig(fig, f"boxplots_{metric}{_part(start, organs_per_page, len(organs))}")
                plt.close(fig)