import os
import pandas as pd

def results_saver(results, output_path):
    """
    Save results dictionary/list to CSV.
    Automatically creates output folders if they do not exist.
    Handles patient physical data (height, weight, BMI) columns.
    Removes duplicate patient columns.
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
    patient_cols = ['patient_id', 'Patient']
    
    # Find the primary patient column (prefer 'patient_id')
    patient_col = None
    for col in patient_cols:
        if col in df.columns:
            patient_col = col
            break
    
    for col in physical_cols:
        if col in df.columns:
            # Forward-fill within each patient group
            if patient_col:
                df[col] = df.groupby(patient_col)[col].ffill().bfill()
            print(f"✅ Processed {col} column")

    # --------------------------------------------------
    # Remove duplicate patient columns (keep only the primary one)
    # --------------------------------------------------
    duplicate_patient_cols = [col for col in patient_cols if col in df.columns and col != patient_col]
    if duplicate_patient_cols:
        df = df.drop(columns=duplicate_patient_cols)
        print(f"🗑️ Removed duplicate patient columns: {duplicate_patient_cols}")

    # --------------------------------------------------
    # Sort alphabetically by patient column
    # --------------------------------------------------
    if patient_col:
        df = df.sort_values(patient_col)

    # --------------------------------------------------
    # Reorder columns: patient → physical data → metrics
    # --------------------------------------------------
    cols = list(df.columns)
    remaining_cols = cols.copy()
    
    # Patient column first
    new_order = []
    if patient_col:
        new_order.append(patient_col)
        remaining_cols.remove(patient_col)
    
    # Physical columns next
    physical_cols_present = [col for col in physical_cols if col in remaining_cols]
    new_order.extend(physical_cols_present)
    for col in physical_cols_present:
        remaining_cols.remove(col)
    
    # Metrics last
    new_order.extend(remaining_cols)
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
    if patient_col:
        print(f"\n📋 Per-patient summary (grouped by '{patient_col}'):")
    for col in physical_cols:
        if col in df.columns:
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                print(f"   {col}: {valid_values.mean():.2f} ± {valid_values.std():.2f} (n={len(valid_values)})")
