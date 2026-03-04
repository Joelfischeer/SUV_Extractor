import os
from pathlib import Path
from tqdm import tqdm as tqdm
import gc

if __name__ == "__main__":

    #Data Directories:
    imgdir = Path("../data/Images/Healthy_Test_Retest_First_Scan")

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

    all_results = []

    for patient in patients:
        pet_folder = imgdir / patient / "STATIC-AC-RECON-LAST-5-MIN"
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
            custom_pet_array=pet_suv  # pass SUV array directly if supported
        )

        # Compute normalized SUV
        patient_result = compute_suv(
            PET_organs,
            organs_of_interest,
            patient_id=patient
        )
        if patient_result is not None:
            all_results.append(patient_result)

    # Save CSV
    results_saver(all_results, output_path)
    print(f"✅ SUV results saved to {output_path}")
        


        
        

        

        
