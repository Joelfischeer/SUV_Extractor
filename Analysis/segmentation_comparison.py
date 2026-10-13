"""
Agreement between MOOSE and TotalSegmentator SUVs.

Per patient both segmentations are applied to the SAME PET (same SUV
conversion), for the organs both tools provide, once without erosion and once
with 1 voxel removed around every mask (3x3x3 structuring element, as in
erode_organ_masks). Compared are SUV_mean and SUV_max.

An organ counts as shared only if it is an exact region in both tools, or if
ALL parts of its combination exist (e.g. MOOSE heart = myocardium + 4
chambers). Substring matches are never used here, because they could compare
different structures (heart vs heart_atrium_left).

Statistics per organ, metric and erosion level (MOOSE - TotalSegmentator):
    n, means, bias (mean difference), SD of differences, Bland-Altman 95%
    limits of agreement, relative bias in % of the pair mean, ICC(A,1)
    (two-way, absolute agreement, single measures), Pearson r, Spearman rho,
    Wilcoxon signed-rank p.
"""

import numpy as np
import pandas as pd

TOOLS = ("totalsegmentator", "moose")
TOOL_LABEL = {"totalsegmentator": "TotalSegmentator", "moose": "MOOSE"}
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


def find_shared_organs(patient_path, organs, combinations, all_shared=False,
                       moose_model_priority=None):
    """
    Organs available in both tools for one patient.

    Parameters
    ----------
    organs : dict backend -> list of organ names (candidates)
    combinations : dict backend -> combination_logic
    all_shared : also compare every region that exists under the same name
        in both tools (ribs, vertebrae, muscles, ...).

    Returns
    -------
    shared : list of organ names
    availability : list of dicts (organ, status per tool) for reporting
    """
    from Image_loading.segmentation_sources import get_segmentation_source

    names = {}
    for tool in TOOLS:
        src = get_segmentation_source(patient_path, tool, None, verbose=False,
                                      moose_model_priority=moose_model_priority)
        names[tool] = {n.lower(): n for n in (src.names() if src else [])}

    candidates = list(dict.fromkeys(
        list(organs["totalsegmentator"]) + list(organs["moose"])
        + list(combinations["totalsegmentator"]) + list(combinations["moose"])))
    if all_shared:
        common = sorted(set(names["totalsegmentator"]) & set(names["moose"]))
        candidates += [names["moose"][n] for n in common
                       if names["moose"][n] not in candidates]

    shared, availability = [], []
    for organ in candidates:
        st = {tool: _status(organ, names[tool], combinations[tool]) for tool in TOOLS}
        availability.append({"organ": organ, **{f"{t}_status": s for t, s in st.items()}})
        if all(s == "ok" for s in st.values()):
            shared.append(organ)
    return shared, availability


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def _organ_values(arr):
    vals = arr[arr > 0]
    if vals.size == 0:
        return np.nan, np.nan, 0
    return float(vals.mean()), float(vals.max()), int(vals.size)


def extract_patient(data_dir, patient, shared, combinations, reference_organ,
                    reference_vertebra, pet_name_hints=None, moose_model_priority=None,
                    erosion_voxels=1):
    """
    SUV_mean / SUV_max per tool, organ and erosion level for one patient.

    Returns (rows, reference_row). rows: list of dicts with
    patient, organ, erosion, tool, suv_mean, suv_max, voxels, erosion_fallback.
    """
    from Image_loading.Image_Loader import PET_Organ_Cropper, erode_organ_masks

    rows, reference = [], {"patient": patient}
    for tool in TOOLS:
        result = PET_Organ_Cropper(
            data_dir=data_dir, patient=patient, organs_of_interest=shared,
            combination_logic=combinations[tool], norm_reference=reference_organ,
            reference_vertebra=reference_vertebra, segmentation_backend=tool,
            pet_name_hints=pet_name_hints, moose_model_priority=moose_model_priority)
        if result is None:
            raise RuntimeError(f"{TOOL_LABEL[tool]} extraction failed")

        reference[f"{tool}_suv_mean"] = result["reference_SUV_mean"]
        reference[f"{tool}_method"] = result["reference_method"]
        reference[f"{tool}_voxels"] = result["reference_voxels"]

        # the reference organ is returned restricted to the vertebra level
        organs = {o: a for o, a in result["organs"].items()
                  if o in shared and o != reference_organ}
        eroded = erode_organ_masks(organs, erosion_config={"DEFAULT": erosion_voxels})

        for organ in shared:
            if organ == reference_organ:
                continue
            for erosion, data in ((0, organs), (erosion_voxels, eroded)):
                if organ in data:
                    mean, mx, vox = _organ_values(data[organ])
                else:
                    mean, mx, vox = np.nan, np.nan, 0
                fallback = False
                if erosion > 0 and organ in organs:
                    # erode_organ_masks keeps a mask uneroded if too little would remain
                    fallback = vox == _organ_values(organs[organ])[2] and vox > 0
                rows.append({"patient": patient, "organ": organ, "erosion": erosion,
                             "tool": tool, "suv_mean": mean, "suv_max": mx,
                             "voxels": vox, "erosion_fallback": fallback})
    return rows, reference


def to_pairs(long_df):
    """Long table -> one row per patient/organ/erosion with both tools side by side."""
    wide = long_df.pivot_table(index=["patient", "organ", "erosion"], columns="tool",
                               values=["suv_mean", "suv_max", "voxels", "erosion_fallback"],
                               aggfunc="first")
    wide.columns = [f"{tool}_{val}" for val, tool in wide.columns]
    wide = wide.reset_index()
    for metric in ("suv_mean", "suv_max"):
        t, m = wide[f"totalsegmentator_{metric}"], wide[f"moose_{metric}"]
        wide[f"diff_{metric}"] = m - t
        wide[f"rel_diff_{metric}_pct"] = 100 * (m - t) / ((m + t) / 2)
    wide["voxel_ratio_moose_ts"] = wide["moose_voxels"] / wide["totalsegmentator_voxels"]
    fb = wide[["totalsegmentator_erosion_fallback", "moose_erosion_fallback"]].fillna(False)
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


def paired_stats(ts, mo):
    """Agreement statistics for paired arrays (MOOSE - TotalSegmentator)."""
    from scipy import stats

    ts, mo = np.asarray(ts, float), np.asarray(mo, float)
    ok = np.isfinite(ts) & np.isfinite(mo)
    ts, mo = ts[ok], mo[ok]
    n = len(ts)
    out = {"n": n}
    if n == 0:
        return out
    d = mo - ts
    rel = 100 * d / ((mo + ts) / 2)
    sd = d.std(ddof=1) if n > 1 else np.nan
    rel_sd = rel.std(ddof=1) if n > 1 else np.nan
    out.update({
        "ts_mean": ts.mean(), "ts_sd": ts.std(ddof=1) if n > 1 else np.nan,
        "moose_mean": mo.mean(), "moose_sd": mo.std(ddof=1) if n > 1 else np.nan,
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


def summarize(pairs, reference_df=None):
    """Summary table: one row per organ x metric x erosion (+ the IVC reference)."""
    rows = []
    for (organ, erosion), grp in pairs.groupby(["organ", "erosion"], sort=False):
        n_fallback = 0
        if erosion > 0:
            n_fallback = int(grp["erosion_fallback"].sum())
            grp = grp[~grp["erosion_fallback"]]      # only truly eroded pairs
        for metric in ("suv_mean", "suv_max"):
            st = paired_stats(grp[f"totalsegmentator_{metric}"], grp[f"moose_{metric}"])
            vr = grp["voxel_ratio_moose_ts"].replace([np.inf, -np.inf], np.nan)
            rows.append({"organ": organ, "metric": metric, "erosion": erosion,
                         "excluded_erosion_fallback": n_fallback,
                         "median_voxel_ratio_moose_ts": vr.median(), **st})
    if reference_df is not None and len(reference_df):
        st = paired_stats(reference_df["totalsegmentator_suv_mean"], reference_df["moose_suv_mean"])
        rows.append({"organ": REFERENCE_LABEL, "metric": "suv_mean", "erosion": "in-plane",
                     "excluded_erosion_fallback": 0, **st})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_comparison(pairs, summary, pdf_path, organs_per_page=16):
    """Multi-page PDF: scatter, Bland-Altman and relative-bias overview."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

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

    with PdfPages(pdf_path) as pdf:
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
                        x, y = g[f"totalsegmentator_{metric}"], g[f"moose_{metric}"]
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
                    ax.set_xlabel(f"TotalSegmentator {metric_label[metric]}", fontsize=8)
                    ax.set_ylabel(f"MOOSE {metric_label[metric]}", fontsize=8)
                    ax.tick_params(labelsize=7)
                axes[0].legend(fontsize=7, loc="upper left")
                fig.suptitle(f"{metric_label[metric]}: MOOSE vs TotalSegmentator (dashed = identity)")
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

                # --- Bland-Altman
                fig, axes = _grid(len(page))
                for ax, organ in zip(axes, page):
                    for e in erosions:
                        g = pairs[(pairs.organ == organ) & (pairs.erosion == e)]
                        if g.empty:
                            continue
                        g = _clean(g)
                        t, m = g[f"totalsegmentator_{metric}"], g[f"moose_{metric}"]
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
                    ax.set_ylabel("MOOSE − TotalSeg", fontsize=8)
                    ax.tick_params(labelsize=7)
                axes[0].legend(fontsize=7, loc="upper left")
                fig.suptitle(f"Bland-Altman {metric_label[metric]} (solid = bias, dotted = 95% limits of agreement)")
                fig.tight_layout()
                pdf.savefig(fig); plt.close(fig)

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
                ax.set_title(f"{metric_label[metric]}: relative bias MOOSE vs TotalSeg (95% LoA)")
                ax.set_xlabel("% of pair mean")
                ax.grid(axis="x", alpha=0.3)
            axes[0].set_yticks(y)
            axes[0].set_yticklabels(organs)
            axes[0].invert_yaxis()
            axes[0].legend()
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)