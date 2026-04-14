# Organ-wise mean Standardized Uptake Value (SUV) Calculation Framework based on PET/CT Images

Framework to calculate mean organ Standardized Uptake Values (SUVs) from PET/CT images. The CT-based organ segmentations are performed by [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

## Getting Started:
### Input:
- Path to a folder where for each individual patient a folder is present. These folder names will end up as IDs in the output table.
- For each patient folder a folder of a DICOM series should be provided for both the PET and according CT image.
- Dictionary with organs of interest. Defined in the SUV_Extractor.py file.

### Output:
- Table with a row per patient and the according mean SUV values for the organs of the patient.

The Segmentation of CT images is based on the TotalSegmentator (https://github.com/wasserth/TotalSegmentator)
