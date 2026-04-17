import os
from pathlib import Path
from tqdm import tqdm as tqdm
import numpy as np

if __name__ == "__main__":

    '''
    Load PET and CT data for a given patient, segmentate organs on CT image, get
    cropped PET volumes along with patient metadata, calculate SUV mean, max and min per organ.
    
    
    Parameters
    -----------
    data_dir : pathlib.Path or str
        Root directory containing all patient folders. Each patient folder is
        expected to include PET and CT DICOM series. The PET and CT folders must
        be named in the following way:

        - A PET folder name containing "PET", "FDG", or "5-MIN"
        - A CT folder name containing "CT"

    organs_of_interest : list of str
        List of organ names to extract from the PET image. These should match
        the available segmentation labels from TotalSegmentator: 
        https://github.com/wasserth/TotalSegmentator?tab=readme-ov-file
        
    combination_logic : dict
        Dictionary defining how multiple segmentation labels should be merged
        into a single organ mask. Keys represent the target organ name, and
        values are lists of label names that will be combined. 
        If an organ is not listed here, it is assumed to correspond to a single
        segmentation label.

    Pipeline Configuration Flags
    ----------------------------

    SEGMENTATE_ORGANS : bool
        If True, performs automatic segmentation of individual organs from CT images. 
        Required for downstream organ-wise PET analysis. The Segmentations are saved in
        the patient folder in new folder called "CT_Segmentation"
        If False, assumes this organ segmentation folder already exists on disk.

    SEGMENTATE_BODY : bool
        If True, generates a whole-body mask from CT.

    SEGMENTATE_LUNG : bool
        If True, performs dedicated lung segmentation using the "lung_nodules" task in the
        TotalSegmentator. Useful for Lung Cancer patients where this semgnetation task performs
        better. Also works for lung lesions.

    NORMALIZE : bool
        If True, normalizes PET intensities using an aorta-based reference
        (blood pool normalization). Each organ’s voxel values are divided by the
        estimated aortic activity, enabling inter-patient comparability.

        Requires valid segmentation of:
            - aorta
            - vertebrae_L1 (used for robust reference estimation)

        If False, raw PET values (e.g., SUV) are used directly.

    ERODE : bool
        If True, applies morphological erosion to organ masks before extracting
        PET values. This removes boundary voxels that may be affected by
        partial volume effects or segmentation inaccuracies.

        The erosion strength is controlled via `erosion_config` This indicates
        the number of voxels to remove in all directions.

        If False, full organ masks are used without modification.


    '''

    data_dir = Path("../../../Data/Clemens_Quadra_lungcancer/Images")
    output_path = Path("../Results")

    SEGMENTATE_ORGANS = True
    SEGMENTATE_BODY = True
    SEGMENTATE_LUNG = True
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
    
    if SEGMENTATE_ORGANS or SEGMENTATE_BODY or SEGMENTATE_LUNG:
        from Image_loading.Segmentator import align_and_segment_images
        align_and_segment_images(
            data_dir=data_dir,
            segmentate_organs_ct=SEGMENTATE_ORGANS,
            segmentate_body_ct=SEGMENTATE_BODY,
            segmentate_lung_ct=SEGMENTATE_LUNG,
            fast=False
        )
    

    #Get Patients in the folder:
    patients = sorted(
    entry.name
    for entry in os.scandir(data_dir)
    if entry.is_dir()
    )

    from Image_loading.Image_Loader import PET_Organ_Cropper
    from Analysis.SUV_calculation import compute_suv
    from Results_saving.results_saver import results_saver
    from Analysis.Normalization_to_aorta import aorta_normalization
    from Image_loading.Image_Loader import erode_organ_masks

    
    all_results = []

    for patient in patients:
        patient_dir = data_dir / patient
        pet_folder = next(
            p for p in patient_dir.iterdir()
            if p.is_dir() and ("LAST-5-MIN" in p.name.upper() or "PET" in p.name.upper())
        )
        
        # 1. Load Cropped organs:
        organs_of_interest_and_L1 = organs_of_interest + ['aorta'] + ['vertebrae_L1']
        result = PET_Organ_Cropper(
            data_dir=data_dir,
            patient=patient,
            organs_of_interest=organs_of_interest_and_L1,
            combination_logic=combination_logic
        )

        # 2. Extract organ data and patient metrics
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

        # 4. Normalize to aorta blood pool:
        if NORMALIZE:

            if "aorta" not in PET_organs_raw or np.sum(PET_organs_raw["aorta"] > 0) == 0:
                print(f"⚠️ No valid aorta for {patient}")
                continue

            if "vertebrae_L1" not in PET_organs_raw or np.sum(PET_organs_raw["vertebrae_L1"] > 0) == 0:
                print(f"⚠️ No valid vertebrae_L1 for {patient}")
                continue

            aorta_reference = aorta_normalization(PET_organs_raw)

            PET_organs_normalized = {}
            for organ, organ_array in PET_organs_raw.items():
                PET_organs_normalized[organ] = organ_array / aorta_reference

        else:
            PET_organs_normalized = {}
            for organ, organ_array in PET_organs_raw.items():
                PET_organs_normalized[organ] = organ_array

        
        # 5. Compute SUVs
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

    results_saver(all_results, output_path)
        
