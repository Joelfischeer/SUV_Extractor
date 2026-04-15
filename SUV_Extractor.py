import os
from pathlib import Path
from tqdm import tqdm as tqdm
import gc
import numpy as np

if __name__ == "__main__":

    #Data Directories, Assumes a folder where for each patient there is a folder with a DICOM series with CT images and one with PET images:
    imgdir = Path("../../Full_Body_Feature_Extraction/data/Images/Healthy_Test_Retest_First_Scan")

    #Output Path:
    output_path = Path("../Results")

    #Do we need to segmentate the images?
    SEGMENTATE_ORGANS = False
    SEGMENTATE_BODY = False
    SEGMENTATE_LUNG = False
    NORMALIZE = False
    ERODE = False


    organs_of_interest = [
        'colon',
        'duodenum',
        'heart',
        'small_bowel',
        'kidneys',
        'liver',
        'pancreas',
        'spleen',
        'stomach',
        'thyroid_gland',
        'brain',
        'muscle',
        'thyroids',
        'heart',
        'superior_vena_cava'
    ]

    translator = {'brain cB': 'brain', 
            'liver cB': 'liver', 
            'pancreas cB': 'pancreas',
            'spleen cB': 'spleen', 
            'stomach cB': 'stomach', 
            'bladder cB': 'bladder',
            'heart cB': 'heart', 
            'intest cB': 'small_intestine',
            'glut cB': 'muscle',
            'Kidneys cB': 
            'kidneys', 'Lungs cB': 'lungs',
            'Thyroids cB': 'thyroids',
            'Adr Gl cB': 'adrenal_glands'}
    
    rename = {#'duodenum': 'small_intestine',
              #'thyroid_gland': 'thyroids'
              }

    combination_logic = {
    'lungs': ['lung_lower_lobe_left', 'lung_middle_lobe_left', 'lung_upper_lobe_left','lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right'],
    'adrenal_glands': ['adrenal_gland_left', 'adrenal_gland_right'],
    'kidneys' : ['kidney_left', 'kidney_right'],
    'muscle': ['gluteus_maximus_right', 'gluteus_maximus_left']
    }


    erosion_config = {
            'aorta': 1,
            'liver': 1,
            'kidneys': 1,
            'spleen': 1,
            'heart': 1,
            'colon': 1,
            'duodenum': 1,
            'small_bowel': 1,
            'stomach': 1,
            'pancreas': 1,
            'thyroids': 1,
            'brain': 1,
            'muscle': 1,
            'heart':1,
            'superior_vena_cava':1,
            'DEFAULT': 1
        }
    


        #If segmentation is needed:
    if SEGMENTATE_ORGANS or SEGMENTATE_BODY or SEGMENTATE_LUNG:
        from Image_loading.Segmentator import align_and_segment_images

        cohort_folders = [p for p in imgdir.iterdir() if p.is_dir()]
        for cohort_folder in cohort_folders:
            align_and_segment_images(
                data_dir=cohort_folder,
                segmentate_organs_ct=SEGMENTATE_ORGANS,
                segmentate_body_ct=SEGMENTATE_BODY,
                segmentate_lung_ct=SEGMENTATE_LUNG,
                fast=False
            )
    

    #Get Patients in the folder:
    patients = sorted(
    entry.name
    for entry in os.scandir(imgdir)
    if entry.is_dir()
    )

    from Image_loading.Image_Loader import PET_Organ_Cropper
    from Analysis.SUV_calculation import compute_suv
    from Results_saving.results_saver import results_saver
    from Analysis.Normalization_to_aorta import aorta_normalization
    from Image_loading.Image_Loader import erode_organ_masks

    
    all_results = []

    for patient in patients:
        patient_dir = imgdir / patient
        pet_folder = next(
            p for p in patient_dir.iterdir()
            if p.is_dir() and ("LAST-5-MIN" in p.name.upper() or "PET" in p.name.upper())
        )
        
        # 2. Load Cropped organs:
        organs_of_interest_and_L1 = organs_of_interest + ['aorta'] + ['vertebrae_L1']
        result = PET_Organ_Cropper(
            data_dir=imgdir,
            patient=patient,
            organs_of_interest=organs_of_interest_and_L1,  # Include aorta and vertebrae L1
            combination_logic=combination_logic
        )
        # Extract organ data and patient metrics
        PET_organs_raw = result["organs"]
        height_m = result["patient_height_m"]
        weight_kg = result["patient_weight_kg"]
        bmi = result["patient_BMI"]
        time_of_scan = result["patient_time_of_scan"]
        
        # 3 Remove pixels around each organ to remove contamination from bad segmentation:
        if ERODE:
            PET_organs_raw = erode_organ_masks(
                PET_organs_raw, 
                erosion_config=erosion_config 
            )

        # 4. Apply renaming:
        #PET_organs_raw = {rename.get(k, k): v for k, v in PET_organs_raw.items()}
        
        # 5. Normalize to aorta blood pool:
        if NORMALIZE:

            if "aorta" not in PET_organs_raw or np.sum(PET_organs_raw["aorta"] > 0) == 0:
                print(f"⚠️ No valid aorta for {patient}")
                continue

            if "vertebrae_L1" not in PET_organs_raw or np.sum(PET_organs_raw["vertebrae_L1"] > 0) == 0:
                print(f"⚠️ No valid vertebrae_L1 for {patient}")
                continue

            aorta_reference = aorta_normalization(PET_organs_raw)

            # Apply normalization to ALL organ crops at once
            PET_organs_normalized = {}
            for organ, organ_array in PET_organs_raw.items():
                PET_organs_normalized[organ] = organ_array / aorta_reference

        else:
            PET_organs_normalized = {}
            for organ, organ_array in PET_organs_raw.items():
                PET_organs_normalized[organ] = organ_array

        
        # 6. Compute final stats (already normalized)
        patient_result = compute_suv(
            PET_organs_normalized,
            organs_of_interest=organs_of_interest + ['aorta'],
            patient_id=patient
        )
        
        if patient_result:
            patient_result['height_m'] = height_m
            patient_result['weight_kg'] = weight_kg
            patient_result['bmi'] = bmi
            patient_result['time_of_scan'] = time_of_scan
            all_results.append(patient_result)

    # Save results
    results_saver(all_results, output_path)
        
