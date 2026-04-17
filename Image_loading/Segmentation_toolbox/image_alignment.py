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

    if os.path.isdir(file_path):
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(file_path)
        reader.SetFileNames(dicom_names)
        img = reader.Execute()
    else:
        img = sitk.ReadImage(file_path)

    direction = np.array(img.GetDirection()).reshape(3, 3)
    standard = np.eye(3)


    if np.allclose(direction, standard, atol=tol):
        is_standard = 1
        return img

    print("Detected non-standard direction matrix:")
    

    # --- Step 1: Detect permutation (axis swaps) ---
    best_perm = None
    for perm in ([0, 1, 2], [1, 0, 2], [0, 2, 1], [2, 0, 1], [1, 2, 0], [2, 1, 0]):
        test = direction[:, perm]
        if np.allclose(np.abs(test), standard, atol=tol):
            best_perm = perm
            break

    if best_perm is not None:
        is_standard = 0
        print(f"Swapping axes according to {best_perm}")
        permuter = sitk.PermuteAxesImageFilter()
        permuter.SetOrder(best_perm)
        img = permuter.Execute(img)

    # --- Step 2: Detect flips (negative directions) ---
    direction = np.array(img.GetDirection()).reshape(3, 3)
    flip_axes = [bool(np.sign(direction[i, i]) < 0) for i in range(3)]

    if any(flip_axes):
        is_standard = 0
        print(f"Flipping axes: {flip_axes}")
        flipper = sitk.FlipImageFilter()
        flipper.SetFlipAxes(flip_axes)
        img = flipper.Execute(img)

    # --- Step 3: Force direction matrix to identity ---
    img.SetDirection(standard.flatten())

    # --- Step 4: Save corrected image ---
    if output_path and is_standard == 0:
        sitk.WriteImage(img, output_path)
        print(f"Corrected image saved to {output_path}")

    return 0

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

