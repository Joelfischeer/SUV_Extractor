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
    from Analysis.SUV_calculation import compute_suv
    from Results_saving.results_saver import results_saver

    all_results = []
    for patient in patients:
        PET_organs = PET_Organ_Cropper(
                        data_dir=imgdir,
                        patient=patient,
                        organs_of_interest=organs_of_interest,
                        combination_logic=combination_logic,
                    )
        
        patient_result = compute_suv(
            PET_organs,
            organs_of_interest,
            patient_id=patient
        )

        if patient_result is not None:
            all_results.append(patient_result)

    results_saver(all_results,
                    output_path)
        


        
        

        

        
