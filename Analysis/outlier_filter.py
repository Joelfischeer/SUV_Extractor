"""
Remove patients with implausible values across many organs.

For every organ (column), each patient's value is compared with the cohort
distribution of that organ:

    z = (value - centre) / spread

method "robust" (default): centre = median, spread = 1.4826 * MAD
    The median absolute deviation is barely affected by the outliers
    themselves. With mean and SD, a few patients with 10x too high values
    inflate the SD so much that they can fall within 2 SD and slip through.
method "sd": centre = mean, spread = SD (classic, sensitive to the outliers)

A patient is excluded if |z| > n_sd in at least min_organs organs. A single
deviating organ (e.g. real disease) never excludes a patient.
Organs with fewer than min_patients values are skipped (no distribution).
"""
import numpy as np
import pandas as pd


def find_outlier_patients(df, value_cols, patient_col="Patient", n_sd=2.0, min_organs=5,
                          method="robust", min_patients=8):
    """
    Returns a DataFrame with one row per patient:
    patient, n_outlier_organs, excluded, outlier_organs (text: organ value
    (cohort centre, z)).
    """
    values = df.set_index(patient_col)[value_cols].apply(pd.to_numeric, errors="coerce")
    z = pd.DataFrame(np.nan, index=values.index, columns=value_cols)
    centres = {}

    for col in value_cols:
        v = values[col].dropna()
        if len(v) < min_patients:
            continue
        if method == "robust":
            centre = v.median()
            spread = 1.4826 * (v - centre).abs().median()
            if spread == 0:                      # many identical values -> fall back to SD
                spread = v.std(ddof=1)
        else:
            centre, spread = v.mean(), v.std(ddof=1)
        if not spread or not np.isfinite(spread):
            continue
        z[col] = (values[col] - centre) / spread
        centres[col] = centre

    flagged = z.abs() > n_sd
    rows = []
    for patient in values.index:
        cols = [c for c in value_cols if flagged.at[patient, c]]
        rows.append({
            "patient": patient,
            "n_outlier_organs": len(cols),
            "n_organs_checked": int(z.loc[patient].notna().sum()),
            "excluded": len(cols) >= min_organs,
            "outlier_organs": "; ".join(
                f"{c.replace('_mean_automatic', '')} {values.at[patient, c]:.3g} "
                f"(cohort {centres[c]:.3g}, z {z.at[patient, c]:+.1f})" for c in cols),
        })
    return pd.DataFrame(rows)


def organ_mean_columns(df):
    """The organ SUV_mean columns of a result table."""
    return [c for c in df.columns if c.endswith("_mean_automatic")]