# Organ-wise mean Standardized Uptake Value (SUV) Calculation Framework based on PET/CT Images

Framework to calculate mean organ Standardized Uptake Values (SUVs) from PET/CT images. The CT-based organ segmentations are performed by [TotalSegmentator](https://github.com/wasserth/TotalSegmentator).

## Getting Started:
### Input:
- Path to a folder where for each individual patient a folder is present. These folder names will end up as IDs in the output table.
- For each patient folder a folder of a DICOM series should be provided for both the PET and according CT image.
- Dictionary with organs of interest. Defined in the SUV_Extractor.py file.

### Output:
- Table with a row per patient and the according mean SUV values for the organs of the patient.


## Calculation of SUVs:

The Script uses the following general formula to calculate for each voxel the SUVs:

```math
\mathrm{SUV}_{bw}(x,y,z)=\frac{C(x,y,z)\cdot W}{D_{\mathrm{corr}}}
```
where 
- $C(x,y,z)$ is the voxel activity concentration in Bq/mL.
- $W$ is the patient body weight in g
- $D_{\mathrm{corr}}$ is the decay-corrected injected Dose in Bq.

### Decay Correction
The $D_{\mathrm{corr}}$ is obtained with the exponential decay function


```math
D_{\mathrm{corr}} = D_0 \, e^{-(\ln 2 / T_{1/2})\Delta t}
```

This equation can then be plugged into the SUV equation. Now we just need to consider two possible cases depending on the PET data.

### BQML Case:
If the original PET image is already provided as an Bq/mL map the units do not have to be converted.
Here the intensity map is already the Activity Concentration.
```math
C(x,y,z)=I(x,y,z)
```
So we end up with the following final formula:

```math
\mathrm{SUV}_{bw}(x,y,z)=\frac{I(x,y,z)\cdot W}{D_0 \, e^{-(\ln 2 / T_{1/2})\Delta t}}
```

### Non BQML Case:
If the orignal PET image is not provided as a Bq/mL map we need to convert it. This is done using a raw PET conversion with 

```math
C(x,y,z)=I(x,y,z)\cdot s+b
```

where 
- $I(x,y,z)$ is the raw voxel intensity value
- $s$ is the RescaleSlope
- $b$ is the RescaleIntercept
- 
Using this formula we end up with the following final formula:

```math
\mathrm{SUV}_{bw}(x,y,z)=\frac{(I(x,y,z)\cdot s+b)\cdot W}{D_0 \, e^{-(\ln 2 / T_{1/2})\Delta t}}
```
