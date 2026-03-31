import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np



def compare_manual_automatic(auto_results_csv, manual_csv, translator, output_path):
    # Load CSVs
    auto_df = pd.read_csv(auto_results_csv, sep=";", decimal=',')
    manual_df = pd.read_csv(manual_csv, decimal=',')
    
    # Convert auto organs to numeric
    auto_organs = [col for col in auto_df.columns if col not in ["Patient"]]
    for col in auto_organs:
        auto_df[col] = pd.to_numeric(auto_df[col], errors='coerce')
    
    rename = {'brain cB': 'brain', 
              'liver cB': 'liver', 
              'pancreas cB': 'pancreas',
              'spleen cB': 'spleen', 
              'stomach cB': 'stomach', 
              'bladder cB': 'bladder',
              'heart cB': 'heart', 
              'intest cB': 'small_intestine', 
              'glut cB': 'muscle',
              'Kidneys cB': 
              'kidneys', 'Lungs cB': 'lungs', ''
              'Thyroids cB': 'thyroids',
              'Adr Gl cB': 'adrenal_glands'}

    # ------------------- FIXED: Convert BEFORE renaming -------------------
    # Keep original manual organ columns + Patient
    manual_organs_raw = list(rename.keys())
    keep_columns_raw = ['Image_ID'] + manual_organs_raw
    manual_df_raw = manual_df[keep_columns_raw].copy()
    
    # Convert original columns to numeric FIRST
    for col in manual_organs_raw:
        if col in manual_df_raw.columns:
            manual_df_raw[col] = pd.to_numeric(manual_df_raw[col], errors='coerce')
    
    # NOW rename safely
    manual_df_clean = manual_df_raw.rename(columns={**rename, 'Image_ID': 'Patient'})

    # ------------------- Create comparison dataframe -------------------
    comparison_df = auto_df[['Patient']].copy()
    common_patients = set(auto_df['Patient']) & set(manual_df_clean['Patient'])
    auto_matched = auto_df[auto_df['Patient'].isin(common_patients)].set_index('Patient')
    manual_matched = manual_df_clean[manual_df_clean['Patient'].isin(common_patients)].set_index('Patient')
    
    for manual_col, auto_col in rename.items():
        if auto_col in auto_df.columns and auto_col in manual_matched.columns:
            comparison_df.loc[comparison_df['Patient'].isin(common_patients), f'{auto_col}_auto'] = auto_matched[auto_col].values
            comparison_df.loc[comparison_df['Patient'].isin(common_patients), f'{auto_col}_manual'] = manual_matched[auto_col].values

    # Melt and plot
    comparison_long = comparison_df.melt(
        id_vars='Patient',
        value_vars=[col for col in comparison_df.columns if '_auto' in col or '_manual' in col],
        var_name='measurement', value_name='SUV'
    )
    comparison_long['organ'] = comparison_long['measurement'].str.replace('_auto|_manual', '', regex=True)
    comparison_long['source'] = comparison_long['measurement'].str.extract('_([a-z]+)$')[0]

    # Boxplot
    plt.figure(figsize=(15, 8))
    sns.boxplot(data=comparison_long, x='organ', y='SUV', hue='source')
    plt.xticks(rotation=45)
    plt.title('Auto vs Manual SUV per Organ')
    plt.tight_layout()
    plt.savefig(f'{output_path}_boxplot.png', dpi=300, bbox_inches='tight')
    plt.close()

    return comparison_df, comparison_long