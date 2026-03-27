import os
from pathlib import Path
from tqdm import tqdm as tqdm
import gc

if __name__ == "__main__":

    #Data Directories:
    imgdir = Path("../../Full_Body_Feature_Extraction/data/Images/Healthy_Test_Retest_First_Scan")
   # imgdir = Path("../Images/Healthy_Test_Retest_First_Scan")    

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
        'thyroid_gland'
    ]

    translator = {
        "Colon": "colon",
        "Duodenum": "duodenum",
        "Heart": "heart",
        "intestine": "small_intestine",
        "Kidneys": "kidneys",
        "Liver": "liver",
        "Pancreas": "pancreas",
        "Spleen": "spleen",
        "Stomach": "stomach",
        "Thyroid": "Thyroids"
    }

    rename = {'small_bowel': 'small_intestine'}

    combination_logic = {
    'lungs': ['lung_lower_lobe_left', 'lung_middle_lobe_left', 'lung_upper_lobe_left','lung_lower_lobe_right', 'lung_middle_lobe_right', 'lung_upper_lobe_right'],
    'adrenal_glands': ['adrenal_gland_left', 'adrenal_gland_right'],
    'kidneys' : ['kidney_left', 'kidney_right']
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

    all_results = []

    for patient in patients:
        patient_dir = imgdir / patient
        pet_folder = next(
            p for p in patient_dir.iterdir()
            if p.is_dir() and ("LAST-5-MIN" in p.name.upper() or "PET" in p.name.upper())
        )
        
        try:
            # 1. Convert to calibrated SUV (Bq/ml → SUV)
            pet_suv_raw = convert_pet_to_suv(pet_folder)  # Raw SUV
            
            # 2. ONE-TIME aorta normalization for entire PET image
            PET_organs_raw = PET_Organ_Cropper(
                data_dir=imgdir,
                patient=patient,
                organs_of_interest=organs_of_interest + ['aorta'],  # Include aorta!
                combination_logic=combination_logic,
                custom_pet_array=pet_suv_raw
            )
            
            if "aorta" not in PET_organs_raw or np.sum(PET_organs_raw["aorta"] > 0) == 0:
                print(f"⚠️ No valid aorta for {patient}")
                continue
                
            # Compute aorta reference ONCE
            aorta_reference = np.mean(PET_organs_raw["aorta"][PET_organs_raw["aorta"] > 0])
            print(f"Aorta reference: {aorta_reference:.3f}")
            
            # 3. Apply normalization to ALL organ crops at once
            PET_organs_normalized = {}
            for organ, organ_array in PET_organs_raw.items():
                PET_organs_normalized[organ] = organ_array / aorta_reference
            
            # 4. Compute final stats (already normalized)
            patient_result = compute_suv(
                PET_organs_normalized,
                organs_of_interest,
                patient_id=patient
            )
            
            if patient_result:
                all_results.append(patient_result)
                
        except Exception as e:
            print(f"❌ Error {patient}: {e}")
            continue

    # Save results
    results_saver(all_results, output_path)

    combined_df = compare_manual_automatic(
    auto_results_csv= f"{output_path}_SUVs.csv",
    manual_csv="../data/Human FDG whole body healthycohort anon.csv",
    translator=translator,
    output_path=output_path
)
        


        
        

        

        
