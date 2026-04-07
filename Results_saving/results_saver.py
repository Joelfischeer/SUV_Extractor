import os
import pandas as pd

def results_saver(results, output_path):
    """
    Save results dictionary/list to CSV.
    Automatically creates output folders if they do not exist.
    Handles patient physical data (height, weight, BMI) columns.
    """

    import os
    import pandas as pd

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

    # --------------------------------------------------
    # Handle patient physical data columns (forward-fill per patient)
    # --------------------------------------------------
    physical_cols = ['height_m', 'weight_kg', 'bmi']
    for col in physical_cols:
        if col in df.columns:
            # Forward-fill within each patient group
            if 'patient_id' in df.columns:
                df[col] = df.groupby('patient_id')[col].ffill().bfill()
            elif 'Patient' in df.columns:
                df[col] = df.groupby('Patient')[col].ffill().bfill()
            print(f"✅ Processed {col} column")

    # --------------------------------------------------
    # Sort alphabetically by patient (if column exists)
    # --------------------------------------------------
    patient_col = next((col for col in ['patient_id', 'Patient'] if col in df.columns), None)
    if patient_col:
        df = df.sort_values(patient_col)

    # --------------------------------------------------
    # Reorder columns: patient info first, then physical data, then metrics
    # --------------------------------------------------
    cols = list(df.columns)
    if patient_col:
        cols.remove(patient_col)
        cols = [patient_col] + cols
    
    physical_cols_present = [col for col in physical_cols if col in cols]
    for col in physical_cols_present:
        cols.remove(col)
    
    new_order = ([patient_col] if patient_col else []) + physical_cols_present + cols
    df = df[new_order]

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------
    csv_path = f'{output_path}_SUVs.csv'
    df.to_csv(csv_path, index=False, sep=';', decimal='.')
    
    print(f"\n✅ SUV results saved to: {csv_path}")
    print(f"📊 Shape: {df.shape}")
    print(f"📈 Columns: {list(df.columns)}")
    
    # Summary stats for physical data
    for col in physical_cols:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                print(f"   {col}: {valid_values.mean():.2f} ± {valid_values.std():.2f} (n={len(valid_values)})")
