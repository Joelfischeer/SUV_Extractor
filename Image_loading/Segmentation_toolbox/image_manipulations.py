import SimpleITK as sitk
import numpy as np



def crop_image_with_segmentation(image, segmentation, image_type, keep_segmentation=True):
    """
    Masks an image or NumPy array using a segmentation mask.

    Parameters
    ----------
    image : sitk.Image or np.ndarray
        The image to be masked (e.g., PET, MRI, CT).
    segmentation : sitk.Image or np.ndarray
        The segmentation mask. Voxels with value > 0 define the mask.
    image_type : str
        'PET', 'CT', or 'MRI'.
    keep_segmentation : bool
        If True, keep voxels where segmentation > 0.
        If False, remove voxels where segmentation > 0 and keep the rest.

    Returns
    -------
    sitk.Image or np.ndarray
        The masked image/array. Returns the same type as the input image.
    """

    # Convert SimpleITK to NumPy if needed
    image_is_itk = isinstance(image, sitk.Image)
    seg_is_itk = isinstance(segmentation, sitk.Image)

    image_np = sitk.GetArrayFromImage(image) if image_is_itk else image
    seg_np = sitk.GetArrayFromImage(segmentation) if seg_is_itk else segmentation

    # Boolean mask
    seg_mask = seg_np > 0

    # Invert mask if we want to REMOVE the segmentation
    if not keep_segmentation:
        seg_mask = ~seg_mask

    # Apply mask depending on modality
    if image_type == 'CT':
        AIR_HU = -1000
        masked_np = image_np.copy()
        masked_np[~seg_mask] = AIR_HU

    elif image_type in ('MRI', 'PET'):
        masked_np = np.zeros_like(image_np)
        masked_np[seg_mask] = image_np[seg_mask]

    else:
        raise ValueError(f"Unsupported image_type: {image_type}")

    # Convert back to ITK if needed
    if image_is_itk:
        masked_img = sitk.GetImageFromArray(masked_np)
        masked_img.CopyInformation(image)
        return masked_img
    else:
        return masked_np
    

import numpy as np


def mean_of_PET_mask(image_array):
    '''
    compute the mean of an image array
    ignore values with zero (only take the mask)

    Parameters
    ----------
    image_array : np.ndarray
        3D image array with shape (z, y, x)

    Returns
    -------
    float

    '''
    mask_nonzero = image_array[image_array > 0]
    return np.mean(mask_nonzero)



def crop_volume_relative_to_segmentation(volume, segmentation, mode):
    """
    Crop a 3D volume above or below a given segmentation.

    Parameters
    ----------
    volume : np.ndarray or sitk.Image
        3D image volume (Z, Y, X). Slice 0 = feet (inferior).
    segmentation : np.ndarray or sitk.Image
        3D binary segmentation mask (same shape as volume).
    mode : str
        "above"  -> keep everything superior to the segmentation
        "below"  -> keep everything inferior to the segmentation

    Returns
    -------
    cropped_volume : np.ndarray
    cropped_segmentation : np.ndarray
    """

    if mode not in ["above", "below"]:
        raise ValueError("mode must be 'above' or 'below'")

    # Convert to numpy if needed
    if isinstance(volume, sitk.Image):
        volume = sitk.GetArrayFromImage(volume)

    if isinstance(segmentation, sitk.Image):
        segmentation = sitk.GetArrayFromImage(segmentation)

    if volume.shape != segmentation.shape:
        raise ValueError("Volume and segmentation must have same shape")

    seg_mask = segmentation > 0

    # Find slices where segmentation exists
    seg_slices = np.where(np.any(seg_mask, axis=(1, 2)))[0]

    if len(seg_slices) == 0:
        raise ValueError("Segmentation not found in volume")

    seg_min = seg_slices.min()  # most inferior slice of segmentation
    seg_max = seg_slices.max()  # most superior slice of segmentation

    if mode == "above":
        # Keep everything superior to segmentation
        cropped_volume = volume[seg_max + 1:]
        cropped_segmentation = segmentation[seg_max + 1:]

    elif mode == "below":
        # Keep everything inferior to segmentation
        cropped_volume = volume[:seg_min]
        cropped_segmentation = segmentation[:seg_min]

    return cropped_volume


def crop_between_segmentations(volume,
                               inferior_segmentation,
                               superior_segmentation):
    """
    Crop volume between two anatomical segmentations.

    Assumes:
        slice index 0 = feet (inferior)
        increasing index = superior

    Parameters
    ----------
    volume : np.ndarray or sitk.Image
    inferior_segmentation : segmentation defining lower boundary (e.g. colon)
    superior_segmentation : segmentation defining upper boundary (e.g. thyroid)

    Returns
    -------
    cropped_volume : np.ndarray
    """

    # Convert to numpy
    if isinstance(volume, sitk.Image):
        volume = sitk.GetArrayFromImage(volume)

    if isinstance(inferior_segmentation, sitk.Image):
        inferior_segmentation = sitk.GetArrayFromImage(inferior_segmentation)

    if isinstance(superior_segmentation, sitk.Image):
        superior_segmentation = sitk.GetArrayFromImage(superior_segmentation)

    if not (volume.shape == inferior_segmentation.shape == superior_segmentation.shape):
        raise ValueError("All inputs must have identical shapes")

    inf_mask = inferior_segmentation > 0
    sup_mask = superior_segmentation > 0

    inf_slices = np.where(np.any(inf_mask, axis=(1, 2)))[0]
    sup_slices = np.where(np.any(sup_mask, axis=(1, 2)))[0]

    if len(inf_slices) == 0 or len(sup_slices) == 0:
        raise ValueError("One of the segmentations is empty")

    # Inferior boundary (colon → closest to feet)
    z_min = inf_slices.min()

    # Superior boundary (thyroid → closest to head)
    z_max = sup_slices.max()

    if z_min >= z_max:
        raise ValueError("Invalid anatomical ordering of segmentations")

    return volume[z_min:z_max + 1]