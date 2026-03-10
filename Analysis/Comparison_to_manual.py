import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



def compare_manual_automatic(auto_results_csv, manual_csv, translator):
    """
    Compare automatic SUV extraction with manual extraction.

    Parameters
    ----------
    auto_results_csv : str or Path
        CSV produced by the automatic pipeline.
    manual_csv : str or Path
        CSV containing manual measurements.
    translator : dict
        Mapping manual organ names -> automatic organ names.
    """

    # -----------------------
    # Load automatic results
    # -----------------------
    auto_df = pd.read_csv(auto_results_csv)

    # -----------------------
    # Load manual results
    # -----------------------
    manual_df = pd.read_csv(manual_csv)

    # Filter only Quadra scan 1
    manual_df = manual_df[
        (manual_df["Scanner"] == "Quadra") &
        (manual_df["scan"] == 1)
    ]

    # -----------------------
    # Rename manual organs
    # -----------------------
    manual_df = manual_df.rename(columns=translator)

    organs = list(translator.values())

    # -----------------------
    # Convert to long format
    # -----------------------
    auto_long = auto_df.melt(
        id_vars=["patient_id"],
        value_vars=organs,
        var_name="organ",
        value_name="SUV"
    )
    auto_long["source"] = "automatic"

    manual_long = manual_df.melt(
        value_vars=organs,
        var_name="organ",
        value_name="SUV"
    )
    manual_long["source"] = "manual"

    # Combine
    combined_df = pd.concat([auto_long, manual_long], ignore_index=True)

    # -----------------------
    # Plot
    # -----------------------
    plt.figure(figsize=(12, 6))

    sns.boxplot(
        data=combined_df,
        x="organ",
        y="SUV",
        hue="source"
    )

    plt.xticks(rotation=45)
    plt.title("Automatic vs Manual SUV Extraction")
    plt.tight_layout()
    plt.show()

    return combined_df