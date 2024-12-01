import pandas as pd
import numpy as np
import nibabel as nib

def downsize_X_v1(movie_name = 'After_The_Rain'):
    
    X_csv_file = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX/'+movie_name+'_exp/Final_df.csv'
    X = pd.read_csv(X_csv_file)
    TR_duration = 1.3
    num_TRs = int(X['Time (s)'].max() // TR_duration)

    # Assign TR indices to each time point
    X['TR'] = (X['Time (s)'] // TR_duration).astype(int)
    # Group by TR and compute the mean for each feature
    X_avg = X.groupby('TR').mean().reset_index()
    # Drop the 'Time (s)' column if not needed
    X_avg = X_avg.drop(columns=['Time (s)','Frame','Scene'], errors='ignore')
    # Select columns whose names start with 'Object', 'Action', or 'Scene'
    columns_to_round = [col for col in X_avg.columns if col.startswith(('Object', 'Action', 'Scene'))]

    # Apply rounding: Set all non-zero values to 1
    X_avg[columns_to_round] = np.where(X_avg[columns_to_round] != 0, 1, 0)
    X_avg = X_avg.drop(columns=['TR'], errors='ignore')

    return X_avg


def setup_Y_v1(X, sub_nb = 2, session = 1, movie_name = 'After_The_Rain'):
    # Assuming fmri_img is a NIfTI image loaded using nibabel
    fmri_img = nib.load("/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Y/"+"sub"+sub_nb+"/"+movie_name+"/"+"smoothed_fmri.nii.gz")
    data = fmri_img.get_fdata()  # Get the image data as a NumPy array

    # Define the number of TRs to remove
    TRs_to_remove = 69

    ##### We start by removing the washed out element from the fmri : #####
    # Step 1: Trim the time points (removing 69 time points from both ends)
    data_trimmed = data[..., TRs_to_remove:-TRs_to_remove]  # Trim the last and first 69 time points

    # Check the shape of the trimmed data
    print(f"Trimmed data shape: {data_trimmed.shape}")
    # Step 2: Adjust the time points to match X_avg
    # Assuming X_avg.shape[0] is the number of time points you want in the trimmed data
    start = int(np.floor((data_trimmed.shape[3] - X.shape[0]) / 2))

    # Step 3: Extract the portion of the data to match the shape of X_avg
    data_weird = data_trimmed[..., start: -(start + 1)]  # Slice to align with X_avg

    # Check the final shape
    print(f"Final trimmed data shape: {data_weird.shape}")

    affine = fmri_img.affine  # Get the affine matrix from the original image

    # Step 1: Create a new NIfTI image with data_weird and the original affine matrix
    new_fmri_img = nib.Nifti1Image(data_weird, affine)
    return new_fmri_img

def setup_Y_fmri(X, sub_nb = 2, session = 1, movie_name = 'After_The_Rain', TR = 1.3, washout = 90):
    # Assuming fmri_img is a NIfTI image loaded using nibabel
    fmri_img = nib.load("/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Y/"+"sub"+str(sub_nb)+"/"+movie_name+"/"+"smoothed_fmri.nii.gz")
    data = fmri_img.get_fdata()  # Get the image data as a NumPy array

    # Define the number of TRs to remove
    TRs_to_remove = int(washout/TR)

    ##### We start by removing the washed out element from the fmri : #####
    # Step 1: Trim the time points (removing 69 time points from both ends)
    data_trimmed = data[..., TRs_to_remove:-TRs_to_remove]  # Trim the last and first 69 time points

    difference = int((data_trimmed.shape[3] - X.shape[0]))

    rectified_Y = data_trimmed[..., 0: -difference]
    # Check the final shape
    print(f"Final trimmed data shape: {rectified_Y.shape}")

    affine = fmri_img.affine  # Get the affine matrix from the original image

    # Step 1: Create a new NIfTI image with data_weird and the original affine matrix
    Y = nib.Nifti1Image(rectified_Y, affine)
    return Y


def setup_Y_npy(X, sub_nb = 2, session = 1, movie_name = 'After_The_Rain', TR = 1.3, washout = 90):
    Y_npy_file = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Y/sub"+str(sub_nb)+"/"+movie_name+"/Y_voxels.npy"
    Y = np.load(Y_npy_file)

    # TRs to remove : 90 / 1.3 = ≈ 69.23

    TRs_to_remove = int(washout/TR)

    #### We start by removing the washed out element from the fmri :
    Y_trimmed = Y[TRs_to_remove:-TRs_to_remove]
    #print(Y_trimmed.shape)

    diff = int(Y_trimmed.shape[0]-X.shape[0])

    Y_weird = Y_trimmed[0: -diff]
    #print(np.floor((Y_trimmed.shape[0]-X_avg.shape[0])/2))
    #print(int(np.floor((Y_trimmed.shape[0]-X_avg.shape[0])/2)))
    print("The shape of Y trimmed :" + Y_weird.shape)

