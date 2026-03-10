import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def compare_manual_automatic(auto_results_csv, manual_csv, translator, output_path):

    # -----------------------
    # Load CSVs
    # -----------------------
    auto_df = pd.read_csv(auto_results_csv, sep=";")
    manual_df = pd.read_csv(manual_csv)

    # -----------------------
    # Filter manual dataset
    # -----------------------
    manual_df = manual_df[
        (manual_df["Scanner"] == "Quadra") &
        (manual_df["scan"] == 1)
    ].copy()

    # -----------------------
    # Match patient IDs
    # -----------------------
    manual_df["Patient"] = manual_df["Patient Name"].str.replace("Quadra", "HTRA")

    # -----------------------
    # Rename organs
    # -----------------------
    manual_df = manual_df.rename(columns=translator)

    # -----------------------
    # Detect common organs
    # -----------------------
    auto_organs = set(auto_df.columns) - {"Patient"}
    manual_organs = set(manual_df.columns)

    organs = sorted(auto_organs.intersection(manual_organs))

    print(f"Using organs: {organs}")

    # -----------------------
    # Long format
    # -----------------------
    auto_long = auto_df.melt(
        id_vars="Patient",
        value_vars=organs,
        var_name="organ",
        value_name="SUV"
    )

    manual_long = manual_df.melt(
        id_vars="Patient",
        value_vars=organs,
        var_name="organ",
        value_name="SUV"
    )

    # -----------------------
    # Pairwise merge
    # -----------------------
    paired = pd.merge(
        auto_long,
        manual_long,
        on=["Patient", "organ"],
        suffixes=("_auto", "_manual")
    )

    # -----------------------
    # Boxplot comparison
    # -----------------------
    combined = pd.concat([
        auto_long.assign(source="automatic"),
        manual_long.assign(source="manual")
    ])

    plt.figure(figsize=(12,6))

    sns.boxplot(
        data=combined,
        x="organ",
        y="SUV",
        hue="source"
    )

    plt.xticks(rotation=45)
    plt.title("Manual vs Automatic SUV Extraction")
    plt.tight_layout()
    plt.savefig(f"{output_path}_Manual_vs_automatic_SUVs_boxplot")
    plt.close()

    # -----------------------
    # Validation figure
    # -----------------------
    unique_organs = paired["organ"].unique()
    n = len(unique_organs)

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))

    results = []

    for i, organ in enumerate(unique_organs):

        subset = paired[paired["organ"] == organ]

        auto = subset["SUV_auto"]
        manual = subset["SUV_manual"]

        r = auto.corr(manual)

        diff = auto - manual
        mean = (auto + manual) / 2
        bias = diff.mean()
        loa_low = bias - 1.96 * diff.std()
        loa_high = bias + 1.96 * diff.std()

        results.append({
            "organ": organ,
            "pearson_r": r,
            "bias_auto_minus_manual": bias,
            "loa_lower": loa_low,
            "loa_upper": loa_high
        })

        # Scatter
        ax = axes[0, i]
        sns.regplot(x=manual, y=auto, ax=ax)

        lims = [
            min(manual.min(), auto.min()),
            max(manual.max(), auto.max())
        ]
        ax.plot(lims, lims, '--')

        ax.set_title(f"{organ}\nr={r:.2f}")
        ax.set_xlabel("Manual SUV")
        ax.set_ylabel("Automatic SUV")

        # Bland-Altman
        ax = axes[1, i]

        ax.scatter(mean, diff)
        ax.axhline(bias, linestyle="--")
        ax.axhline(loa_low, linestyle=":")
        ax.axhline(loa_high, linestyle=":")

        ax.set_title(f"{organ}\nBias={bias:.2f}")
        ax.set_xlabel("Mean SUV")
        ax.set_ylabel("Auto − Manual")

    plt.tight_layout()
    plt.savefig(f"{output_path}_Manual_vs_automatic_SUVs_correlation")
    plt.close()

    stats_df = pd.DataFrame(results)

    print("\nPairwise statistics:")
    print(stats_df)

    return combined, paired, stats_df