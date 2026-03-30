import os
from pathlib import Path
from tqdm import tqdm as tqdm
import gc
import numpy as np

if __name__ == "__main__":

    #Data Directories:
    #imgdir = Path("../../Full_Body_Feature_Extraction/data/Images/Healthy_Test_Retest_First_Scan")
    imgdir = Path("../Images/Healthy_Test_Retest_First_Scan")    

    output_path = Path("../Results/Healthy_Test_Retest_First_Scan")

    #Do we need to segmentate the images?
    SEGMENTATE = False


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
        'thyroids'
    ]

    translator = {
        "colon cb": "colon",
        "duodenum cb": "duodenum",
        "heart cB": "heart",
        "intest cB": "small_intestine",
        "Kidneys cB": "kidneys",
        "liver cB": "liver",
        "pancreas cB": "pancreas",
        "spleen cB": "spleen",
        "stomach cB": "stomach",
        "Thyroids cB": "thyroids",
        'brain cB': 'brain',
        'Lungs cB': 'lungs',
        'glut cB': 'muscle',
        'bladder cB': 'bladder',
        'fat visc cB': 'fat'
    }

    rename = {'small_bowel': 'small_intestine',
              'thyroid_gland': 'thyroids'}

    combination_logic = {
    'lungs': ['lung_lower_lobe_left', 'lung_middle_lobe_left', 'lung_upper_lobe_left','lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right'],
    'adrenal_glands': ['adrenal_gland_left', 'adrenal_gland_right'],
    'kidneys' : ['kidney_left', 'kidney_right'],
    'muscle': ['gluteus_maximus_right', 'gluteus_maximus_left']
    }


    erosion_config = {
            'aorta': 1,
            'liver': 2,
            'kidneys': 2,
            'spleen': 2,
            'heart': 2,
            'colon': 3,
            'duodenum': 3,
            'small_bowel': 3,
            'stomach': 3,
            'pancreas': 3,
            'thyroid_gland': 2,
            'thyroids': 2,
            'brain': 2,
            'muscle': 1,
            'DEFAULT': 2
        }
    


        #If segmentation is needed:
    if SEGMENTATE == True:
        from Image_loading.Segmentator import align_and_segment_images

        #Align + Segment
        align_and_segment_images(
            data_dir=imgdir,
            segmentate_organs_ct= SEGMENTATE,
            segmentate_body_ct= False,
            fast= False
        )

    #Get Patients in the folder:
    patients = sorted(
    entry.name
    for entry in os.scandir(imgdir)
    if entry.is_dir()
    )

    from Image_loading.Image_Loader import PET_Organ_Cropper
    from Analysis.SUV_calculation import compute_suv, convert_pet_to_suv
    from Results_saving.results_saver import results_saver
    from Analysis.Comparison_to_manual import compare_manual_automatic
    from Analysis.Normalization_to_aorta import aorta_normalization
    from Image_loading.Image_Loader import erode_organ_masks

    all_results = []

    for patient in patients:
        patient_dir = imgdir / patient
        pet_folder = next(
            p for p in patient_dir.iterdir()
            if p.is_dir() and ("LAST-5-MIN" in p.name.upper() or "PET" in p.name.upper())
        )
        

        # 1. Convert to calibrated SUV (Bq/ml → SUV)
        pet_suv_raw = convert_pet_to_suv(pet_folder)  # Raw SUV
        
        # 2. Load Cropped organs:
        organs_of_interest_and_L1 = organs_of_interest + ['aorta'] + ['vertebrae_L1']
        PET_organs_raw = PET_Organ_Cropper(
            data_dir=imgdir,
            patient=patient,
            organs_of_interest=organs_of_interest_and_L1,  # Include aorta and vertebrae L1
            combination_logic=combination_logic,
            custom_pet_array=pet_suv_raw
        )
        
        # 3 Remvove pixels around each organ to remove contamination from bad segmentation:
        PET_organs_raw = erode_organ_masks(
            PET_organs_raw, 
            erosion_config=erosion_config 
        )

        # 4. Apply renaming:
        PET_organs_raw = {rename.get(k, k): v for k, v in PET_organs_raw.items()}

        if "aorta" not in PET_organs_raw or np.sum(PET_organs_raw["aorta"] > 0) == 0:
            print(f"⚠️ No valid aorta for {patient}")
            continue

        if "vertebrae_L1" not in PET_organs_raw or np.sum(PET_organs_raw["vertebrae_L1"] > 0) == 0:
            print(f"⚠️ No valid vertebrae_L1 for {patient}")
            continue
        
        # 5. Compute aorta reference
        aorta_reference = aorta_normalization(PET_organs_raw)
        print(aorta_reference)

        # 6. Apply normalization to ALL organ crops at once
        PET_organs_normalized = {}
        for organ, organ_array in PET_organs_raw.items():
            PET_organs_normalized[organ] = organ_array / aorta_reference
        
        # 7. Compute final stats (already normalized)
        patient_result = compute_suv(
            PET_organs_normalized,
            organs_of_interest=organs_of_interest + ['aorta'],
            patient_id=patient
        )
        
        if patient_result:
            all_results.append(patient_result)
                

    # Save results
    results_saver(all_results, output_path)

    combined_df = compare_manual_automatic(
    auto_results_csv= f"{output_path}_SUVs.csv",
    manual_csv="../data/Quadra_test_retest.csv",
    translator=translator,
    output_path=output_path
)
        


        
        

        

        
