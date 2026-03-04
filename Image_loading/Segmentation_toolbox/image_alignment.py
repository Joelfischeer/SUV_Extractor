import numpy as np
import SimpleITK as sitk
import os

def align_image_orientation(file_path, output_path=None, tol=1e-4):
    """
    Automatically correct image orientation by detecting necessary axis swaps or flips.
    Works for both flipped and mirrored (non-diagonal) direction matrices.

    Parameters
    ----------
    file_path : str
        Path to the DICOM folder or NIfTI file.
    output_path : str, optional
        Where to save the corrected image (.nii.gz).
    tol : float, optional
        Tolerance for floating-point comparison.
    """
    # --- Read image (DICOM or NIfTI) ---
    if os.path.isdir(file_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_path)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(file_path)

    direction = np.array(img.GetDirection()).reshape(3, 3)
    standard = np.eye(3)

    # If already standard
    if np.allclose(direction, standard, atol=tol):
        is_standard = 1
        return img

    print("⚙️ Detected non-standard direction matrix:")
    

    # --- Step 1: Detect permutation (axis swaps) ---
    best_perm = None
    for perm in ([0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 0, 1], [1, 2, 0], [2, 1, 0]):
        test = direction[:, perm]
        if np.allclose(np.abs(test), standard, atol=tol):
            best_perm = perm
            break

    if best_perm is not None:
        is_standard = 0
        print(f"🔁 Swapping axes according to {best_perm}")
        permuter = sitk.PermuteAxesImageFilter()
        permuter.SetOrder(best_perm)
        img = permuter.Execute(img)

    # --- Step 2: Detect flips (negative directions) ---
    direction = np.array(img.GetDirection()).reshape(3, 3)
    flip_axes = [bool(np.sign(direction[i, i]) < 0) for i in range(3)]

    if any(flip_axes):
        is_standard = 0
        print(f"🔄 Flipping axes: {flip_axes}")
        flipper = sitk.FlipImageFilter()
        flipper.SetFlipAxes(flip_axes)
        img = flipper.Execute(img)

    # --- Step 3: Force direction matrix to identity ---
    img.SetDirection(standard.flatten())

    # --- Step 4: Save corrected image ---
    if output_path and is_standard == 0:
        sitk.WriteImage(img, output_path)
        print(f"✅ Corrected image saved to {output_path}")

    return 0



def compare_geometry(img1, img2, tol=1e-3):
    """
    Compare the geometric properties of two SimpleITK images.
    So in essence check if they are aligned

    
    Parameters
    ----------------------
    Input: Two SimpleITK.Images

    Output: dictionary
        - 'origin_aligned'
        - 'spacing_aligned'
        - 'direction_aligned'
        - 'all_aligned'
        - plus the actual values for reference

    """

    # Compute individual checks
    same_origin = np.allclose(img1.GetOrigin(), img2.GetOrigin(), atol=tol)
    same_spacing = np.allclose(img1.GetSpacing(), img2.GetSpacing(), atol=tol)
    same_direction = np.allclose(
        np.array(img1.GetDirection()).reshape(3,3),
        np.array(img2.GetDirection()).reshape(3,3),
        atol=tol
    )

    # Combine results
    results = {
        "origin_aligned": same_origin,
        "spacing_aligned": same_spacing,
        "direction_aligned": same_direction,
        "all_aligned": same_origin and same_spacing and same_direction,
        "origin_diff": np.array(img1.GetOrigin()) - np.array(img2.GetOrigin()),
        "spacing_diff": np.array(img1.GetSpacing()) - np.array(img2.GetSpacing()),
        "direction_diff": np.array(img1.GetDirection()).reshape(3,3) - np.array(img2.GetDirection()).reshape(3,3)
    }

    # Print human-readable summary
    print("🔍 Geometry comparison:")
    print(f"  Origin aligned?     {same_origin}")
    print(f"    A origin:  {img1.GetOrigin()}")
    print(f"    B origin: {img2.GetOrigin()}")
    print(f"    Δ Origin:      {results['origin_diff']}\n")

    print(f"  Spacing aligned?    {same_spacing}")
    print(f"    A spacing:  {img1.GetSpacing()}")
    print(f"    B spacing: {img2.GetSpacing()}")
    print(f"    Δ Spacing:      {results['spacing_diff']}\n")

    print(f"  Direction aligned?  {same_direction}")
    if not same_direction:
        print("    Direction difference matrix:\n", results["direction_diff"])

    print(f"\n✅ All geometry aligned? {results['all_aligned']}")
    return results




def align_image(image_to_align, reference_image, align_origin=True, align_spacing=True, align_direction=True, interpolation=sitk.sitkLinear):

    """
    Align a moving SimpleITK image to a fixed image.

    Parameters
    ----------
    image_to_align : SimpleITK.Image
        The image to align.
    reference_image : SimpleITK.Image
        The reference image.
    align_origin : bool
        If True, align the origin.
    align_spacing : bool
        If True, align voxel spacing.
    align_direction : bool
        If True, align image orientation.
    interpolation : SimpleITK interpolator
        Interpolation method to use (e.g., sitk.sitkLinear, sitk.sitkNearestNeighbor).

    Returns
    -------
    SimpleITK.Image
        Aligned moving image.
    """

    # Create a copy to avoid modifying the original
    moving_aligned = sitk.Image(image_to_align)

    # Prepare reference image with only the components we want to align
    ref_size = image_to_align.GetSize()
    ref_spacing = image_to_align.GetSpacing()
    ref_origin = image_to_align.GetOrigin()
    ref_direction = image_to_align.GetDirection()

    #Check the boolean inputs on what needs to be aligned
    if align_spacing:
        ref_spacing = reference_image.GetSpacing()
    if align_origin:
        ref_origin = reference_image.GetOrigin()
    if align_direction:
        ref_direction = reference_image.GetDirection()

    # Create a reference image with the fixed components
    reference = sitk.Image(
        reference_image.GetSize(), # sets the number of voxels in each dimension to match the reference image
        image_to_align.GetPixelID() # sets the pixel type to match the image_to_align (avoid type conflicts) 
    )

    reference.SetSpacing(ref_spacing)
    reference.SetOrigin(ref_origin)
    reference.SetDirection(ref_direction)

    # Resample moving image onto the reference
    resampled = sitk.Resample(
        moving_aligned,
        reference,
        sitk.Transform(),  # identity transform
        interpolation,
        0.0,
        image_to_align.GetPixelID()
    )

    return resampled


import SimpleITK as sitk
import numpy as np


def resample_to_spacing(image, target_spacing=(4.0, 4.0, 3.0), is_label=False):
    """
    Resample a SimpleITK image to target spacing.

    Parameters
    ----------
    image : sitk.Image
    target_spacing : tuple
        Desired voxel spacing (x,y,z)
    is_label : bool
        If True -> nearest neighbor interpolation (for masks)
        If False -> linear interpolation (for CT/PET)

    Returns
    -------
    sitk.Image
    """

    original_spacing = image.GetSpacing()
    original_size = image.GetSize()

    new_size = [
        int(round(original_size[i] * (original_spacing[i] / target_spacing[i])))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    return resampler.Execute(image)


import os
import numpy as np
import SimpleITK as sitk
import gc
from tqdm import tqdm
import random

def compute_target_organ_sizes(imgdir,
                               cohorts,
                               organs_of_interest,
                               combination_logic,
                               percentile=95,
                               sample_per_cohort=5):
    """
    Computes per-organ target sizes based on bounding boxes around 1s in segmentations.
    Samples a limited number of patients per cohort for speed.

    Parameters
    ----------
    imgdir : str or Path
        Root directory of image cohorts.
    cohorts : list of str
        List of cohort folder names to sample from.
    organs_of_interest : list
        List of organ names.
    combination_logic : dict
        Organ combination mapping, used to identify combined organs.
    percentile : float
        Percentile of organ dimensions to use as target size (default 95th).
    sample_per_cohort : int
        Number of random patients to sample per cohort.

    Returns
    -------
    dict
        Dictionary mapping organ -> target size (z, y, x)
    """
    organ_shapes = {organ: [] for organ in organs_of_interest}
    print("\n=== Computing per-organ target sizes from bounding boxes ===")

    for cohort in cohorts:
        cohort_path = os.path.join(imgdir, cohort)
        if not os.path.exists(cohort_path):
            print(f"⚠️ Cohort folder not found: {cohort}")
            continue

        all_patients = [p for p in os.listdir(cohort_path) if os.path.isdir(os.path.join(cohort_path, p))]
        if not all_patients:
            print(f"⚠️ No patients found in cohort {cohort}")
            continue

        # Sample patients randomly
        sampled_patients = random.sample(all_patients, min(sample_per_cohort, len(all_patients)))

        for patient in tqdm(sampled_patients, desc=f"Processing cohort {cohort}"):
            patient_path = os.path.join(cohort_path, patient)
            seg_folder = os.path.join(patient_path, "CT_segmentation")
            if not os.path.exists(seg_folder):
                print(f"⚠️ No segmentation folder for {cohort}/{patient}, skipping.")
                continue

            # Load segmentations
            available_segs = {}
            for seg_file in os.listdir(seg_folder):
                seg_path = os.path.join(seg_folder, seg_file)
                try:
                    seg_img = sitk.ReadImage(seg_path)
                    available_segs[seg_file] = seg_img
                except Exception as e:
                    print(f"⚠️ Could not read {seg_path}: {e}")

            # Combine organs
            if combination_logic is not None:
                for combined_name, components in combination_logic.items():
                    combined_mask = None
                    for comp in components:
                        matched_file = next((f for f in available_segs.keys() if comp.lower() in f.lower()), None)
                        if matched_file:
                            part_mask = available_segs[matched_file]
                            combined_mask = part_mask if combined_mask is None else sitk.Or(combined_mask, part_mask)
                    if combined_mask is not None:
                        available_segs[combined_name] = combined_mask

            # Compute bounding boxes
            for organ in organs_of_interest:
                matched_key = next((k for k in available_segs.keys() if organ.lower() in k.lower()), None)
                if matched_key:
                    seg_np = sitk.GetArrayFromImage(available_segs[matched_key])
                    non_zero_indices = np.argwhere(seg_np > 0)
                    if non_zero_indices.size == 0:
                        continue
                    min_z, min_y, min_x = non_zero_indices.min(axis=0)
                    max_z, max_y, max_x = non_zero_indices.max(axis=0)
                    bbox_shape = (max_z - min_z + 1, max_y - min_y + 1, max_x - min_x + 1)
                    organ_shapes[organ].append(bbox_shape)

            del available_segs
            gc.collect()

    # Compute target sizes
    target_sizes = {}
    print("\n=== Organ target sizes (based on percentile of bounding boxes) ===")
    for organ, shapes in organ_shapes.items():
        if not shapes:
            print(f"⚠️ No shapes found for {organ}, skipping.")
            continue
        shapes_np = np.array(shapes)
        organ_target = np.percentile(shapes_np, percentile, axis=0)
        organ_target = tuple([int(np.ceil(s)) for s in organ_target])
        target_sizes[organ] = organ_target
        print(f"  {organ}: target size {organ_target} (based on {len(shapes)} samples)")

    return target_sizes
