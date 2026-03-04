import os
import pandas as pd

def results_saver(results, output_path):
    """
    Save results dictionary/list to CSV.
    Automatically creates output folders if they do not exist.
    """

    # --------------------------------------------------
    # Create directory if it does not exist
    # --------------------------------------------------
    output_dir = os.path.dirname(output_path)

    if output_dir != "":
        os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------
    # Convert to DataFrame
    # --------------------------------------------------
    df = pd.DataFrame(results)

    # Sort alphabetically by patient (if column exists)
    if "Patient" in df.columns:
        df = df.sort_values("Patient")

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    df.to_csv(f'{output_path}_SUVs.csv', index=False, sep=';')

    print(f"\n✅ SUV results saved to: {output_path}")
