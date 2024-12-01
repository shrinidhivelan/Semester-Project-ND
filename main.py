from preprocessing import *

def main():
    output_folder = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Y'
    subject_number = 2
    movie_name = "After_The_Rain"

    print(f"Smoothing the fmri data for subject number {subject_number} ")
    #smoothen_fmri(subject_number = subject_number, movie_name = movie_name, output_folder = output_folder, file_name = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/irmf/derivatives/preprocessing/sub-S02/ses-1/func/sub-S02_ses-1_task-AfterTheRain_space-MNI_desc-ppres_bold.nii.gz')
    
    print(f"Creating brain mask...")
    brain_mask = create_brain_mask(subject_number, output_folder, movie_name, "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/irmf/derivatives/preprocessing/sub-S02/ses-2/anat/sub-S02_ses-2_space-subject_desc-anat_bold.nii.gz")
    
    print(f"Creating Y matrix for GLM...")
    Y = create_Y_voxel_wise(2, output_folder, movie_name)

    #np.save('Y_voxel_time_series.npy', Y)
    np.save(f'/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Y/sub{subject_number}/{movie_name}/Y_voxels.npy',Y)


if __name__ == "__main__":
    main()