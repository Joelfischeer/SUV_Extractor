Script to Calculate mean Standardized Uptake Values (SUVs) from PET/CT images. As input a path to a folder should be provided where for each individual patient a folder is present. 
These folder names will end up as IDs in the output table. For each patient a folder of a DICOM series should be provided for both the PET and according CT image. The output
is a table with a row per patient and the according mean SUV values for the organs of the patient. The organs of interest can be defined in the SUV_Extractor.py file.

The Segmentation of CT images is based on the TotalSegmentator (https://github.com/wasserth/TotalSegmentator)
