import nibabel as nib
import matplotlib.pyplot as plt
from nilearn.image import smooth_img
from nilearn.plotting import plot_stat_map
import os

from nilearn.input_data import NiftiLabelsMasker, NiftiMasker
from nilearn import datasets, image, masking
import numpy as np
import nibabel as nib


def smoothen_fmri(subject_number = 2, movie_name = 'Movie', output_folder = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Y', file_name = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/irmf/derivatives/preprocessing/sub-S02/ses-1/func/sub-S02_ses-1_task-AfterTheRain_space-MNI_desc-ppres_bold.nii.gz'):
    save_repo = f"{output_folder}/sub{subject_number}/{movie_name}"

    os.makedirs(save_repo, exist_ok=True)
    fmri_img = nib.load(file_name)
    # Specify the FWHM in millimeters (e.g., 6mm)
    fwhm = 6

    # Smooth the fMRI image
    smoothed_img = smooth_img(fmri_img, fwhm)

    # Save the smoothed image to a new file
    smoothed_img.to_filename(f"{save_repo}/smoothed_fmri.nii.gz")


def create_brain_mask(subject_nb = 2, output_folder='', movie_name="", mri_image_path = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/irmf/derivatives/preprocessing/sub-S02/ses-2/anat/sub-S02_ses-2_space-subject_desc-anat_bold.nii.gz"):
    
    save_repo = f"{output_folder}/sub{subject_nb}/{movie_name}"

    # Load your anatomical MRI image (replace the file path as needed)
    print("Loading anatomical image....")
    anat_img = image.load_img(mri_image_path)

    # Create a brain mask using nilearn's masking function
    print("Creating brain mask...")
    brain_mask = masking.compute_brain_mask(anat_img)

    # Check the shape of the brain mask
    print(f"Brain mask shape: {brain_mask.shape}")

    # Save the brain mask to a file (optional)
    brain_mask.to_filename(save_repo+'/brain_mask.nii.gz')
    return brain_mask

def create_Y_voxel_wise(subject_nb = 2, output_folder='', movie_name=""):
    save_repo = f"{output_folder}/sub{subject_nb}/{movie_name}"
    brain_mask = nib.load(f'{save_repo}/brain_mask.nii.gz')
    
    # Load the full brain or selected voxel data
    full_fmri_data = f'{save_repo}/smoothed_fmri.nii.gz'  # This should be a 4D fMRI image
    fmri_img = nib.load(full_fmri_data)

    print("resampled the brain mask!")
    # Resample the brain mask to match the fMRI image's spatial dimensions
    resampled_brain_mask = image.resample_to_img(brain_mask, fmri_img, interpolation='nearest')

    print("Creating the brain masker...")
    masker = NiftiMasker(mask_img=resampled_brain_mask, standardize=True)

    print("Fit the masker...")
    voxel_time_series = masker.fit_transform(full_fmri_data)

    # Y would be voxel_time_series here
    return voxel_time_series




