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
        #Get the folder with the PET DISOMs inside
        patient_dir = imgdir / patient
        pet_folder = next(
            p for p in patient_dir.iterdir()
            if p.is_dir() and ("LAST-5-MIN" in p.name.upper() or "PET" in p.name.upper())
        )
        #Convert to SUV image
        try:
            pet_suv = convert_pet_to_suv(pet_folder)
        except Exception as e:
            print(f"⚠️ Could not load PET for {patient}: {e}")
            continue
        
        # Crop organs (returns dict of organ_name -> np.array of SUV values)
        PET_organs = PET_Organ_Cropper(
            data_dir=imgdir,
            patient=patient,
            organs_of_interest=organs_of_interest,
            combination_logic=combination_logic,
            custom_pet_array=pet_suv  
        )

        # Compute normalized SUV
        patient_result = compute_suv(
            PET_organs,
            organs_of_interest,
            patient_id=patient
        )

        #Rename the columns which the TotalSegmentator names differently:
        patient_result = {rename.get(k, k): v for k, v in patient_result.items()}

        #Store the results:
        if patient_result is not None:
            all_results.append(patient_result)

    # Save CSV
    results_saver(all_results, output_path)

    combined_df = compare_manual_automatic(
    auto_results_csv= f"{output_path}_SUVs.csv",
    manual_csv="../data/Human FDG whole body healthycohort anon.csv",
    translator=translator,
    output_path=output_path
)
        


        
        

        

        
