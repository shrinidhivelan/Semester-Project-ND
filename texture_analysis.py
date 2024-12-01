import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import os

def analyze_video_texture(video_path, save_path_dataframe, save_path_image, movie_name='Movie'):
    """
    Analyze video texture characteristics using GLCM and extract contrast, correlation, energy, and homogeneity.
    Save the results as a CSV and plot the time series of these characteristics.
    
    Args:
        video_path (str): Path to the input MP4 video file.
        save_path_dataframe (str): Directory to save the DataFrame as CSV.
        save_path_image (str): Directory to save the plot image.
        movie_name (str): Name of the movie for naming the saved files.
    """
    # Step 1: Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Step 2: Initialize lists to store values
    contrast_values = []
    correlation_values = []
    energy_values = []
    homogeneity_values = []
    frame_count = 0

    # Get FPS and validate
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30  # Default FPS if the value is invalid

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for texture analysis
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 3: Calculate GLCM (Gray Level Co-occurrence Matrix)
        glcm = graycomatrix(gray_frame, distances=[1], angles=[0], symmetric=True, normed=True)

        # Step 4: Extract texture features from GLCM
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

        # Append the features for each frame
        contrast_values.append(contrast)
        correlation_values.append(correlation)
        energy_values.append(energy)
        homogeneity_values.append(homogeneity)

        frame_count += 1

    cap.release()

    # Step 5: Prepare the time series
    time = np.arange(frame_count) / fps  # Convert frame indices to time (seconds)

    # Create a directory structure for saving files
    os.makedirs(f"{save_path_dataframe}/{movie_name}", exist_ok=True)
    os.makedirs(f"{save_path_image}/{movie_name}", exist_ok=True)

    # Step 6: Save texture features to a CSV
    texture_data = {
        "Frame Index": np.arange(frame_count),
        "Time (s)": time,
        "Contrast": contrast_values,
        "Correlation": correlation_values,
        "Energy": energy_values,
        "Homogeneity": homogeneity_values
    }
    texture_df = pd.DataFrame(texture_data)
    texture_df.to_csv(f"{save_path_dataframe}/{movie_name}/texture_analysis_data.csv", index=False)
    print(f"Texture analysis data saved as '{save_path_dataframe}/{movie_name}/texture_analysis_data.csv'.")

    # Step 7: Plot the results
    plt.figure(figsize=(12, 8))

    # Plot the contrast time series
    plt.subplot(4, 1, 1)
    plt.plot(time, contrast_values, label='Contrast over Time', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Contrast')
    plt.title('Contrast Time Series')
    plt.legend()

    # Plot the correlation time series
    plt.subplot(4, 1, 2)
    plt.plot(time, correlation_values, label='Correlation over Time', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Correlation')
    plt.title('Correlation Time Series')
    plt.legend()

    # Plot the energy time series
    plt.subplot(4, 1, 3)
    plt.plot(time, energy_values, label='Energy over Time', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Energy')
    plt.title('Energy Time Series')
    plt.legend()

    # Plot the homogeneity time series
    plt.subplot(4, 1, 4)
    plt.plot(time, homogeneity_values, label='Homogeneity over Time', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Homogeneity')
    plt.title('Homogeneity Time Series')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path_image}/{movie_name}/texture_features_plot.png")
    print(f"Texture features plot saved as '{save_path_image}/{movie_name}/texture_features_plot.png'.")


""" 
video_path = '/Users/shrinidhivelan/Documents/Projet CSE I/data/FilmFiles/Sintel_exp.mp4'
save_path_dataframe = '/Users/shrinidhivelan/Documents/TextureData'
save_path_image = '/Users/shrinidhivelan/Documents/TexturePlots'
analyze_video_texture(video_path, save_path_dataframe, save_path_image, movie_name='Sintel_exp')
"""
