import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
import os

def analyze_video_characteristics(video_path, save_path_dataframe, save_path_image, movie_name = 'Sintel_exp'):
    """
    Analyze video characteristics such as brightness, contrast, and sharpness over time.
    Save the results and frequency spectrum of brightness to CSV files and plot the results.
    
    Args:
        video_path (str): Path to the input MP4 video file.
        save_path (str): Directory path to save the output CSV files.
    """
    # Step 1: Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Step 2: Initialize lists to store values
    brightness_values = []
    contrast_values = []
    sharpness_values = []
    frame_count = 0

    # Get FPS and check validity
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30  # Default FPS if the value is invalid

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to grayscale for contrast and sharpness calculations
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate average brightness using the V (Value) channel
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv_frame[:, :, 2])  # Use the V (brightness) channel
        brightness_values.append(avg_brightness)

        # Calculate contrast as the standard deviation of pixel intensities
        contrast = np.std(gray_frame)
        contrast_values.append(contrast)

        # Calculate sharpness using the Laplacian operator (variance of Laplacian)
        sharpness = cv2.Laplacian(gray_frame, cv2.CV_64F).var()
        sharpness_values.append(sharpness)

        frame_count += 1

    cap.release()

    # Step 3: Prepare the time series
    brightness_values = np.array(brightness_values)
    contrast_values = np.array(contrast_values)
    sharpness_values = np.array(sharpness_values)

    time = np.arange(frame_count) / fps  # Convert frame indices to time (seconds)

    # Step 4: Fourier Transform for brightness
    N = len(brightness_values)
    fft_values = fft(brightness_values)
    frequencies = fftfreq(N, d=1 / fps)  # Sampling interval is 1/FPS

    # Get positive frequencies and magnitudes for brightness
    positive_freqs = frequencies[:N // 2]
    positive_magnitudes = np.abs(fft_values[:N // 2])

    # Step 5: Save table with frame index, time, brightness, contrast, and sharpness
    table_data = {
        "Frame Index": np.arange(frame_count),
        "Time (s)": time,
        "Brightness Value": brightness_values,
        "Contrast Value": contrast_values,
        "Sharpness Value": sharpness_values
    }
    df = pd.DataFrame(table_data)
    os.makedirs(save_path_dataframe+"/"+movie_name, exist_ok=True)
    df.to_csv(f"{save_path_dataframe}/{movie_name}/brightness_contrast_sharpness_data.csv", index=False)


    # Save frequency spectrum data for brightness to CSV
    freq_table = {
        "Frequency (Hz)": positive_freqs,
        "Magnitude": positive_magnitudes
    }
    freq_df = pd.DataFrame(freq_table)
    freq_df.to_csv(f"{save_path_dataframe}/{movie_name}/frequency_spectrum_data.csv", index=False)

    print(f"Tables saved as '{save_path_dataframe}/{movie_name}/video_characteristics_data.csv' and '{save_path_dataframe}/{movie_name}/frequency_spectrum_data.csv'.")

    # Step 6: Plot the results
    plt.figure(figsize=(12, 8))

    # Plot the brightness time series
    plt.subplot(3, 1, 1)
    plt.plot(time, brightness_values, label='Brightness over Time', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('Average Brightness')
    plt.title('Brightness Time Series')
    plt.legend()

    # Plot the contrast time series
    plt.subplot(3, 1, 2)
    plt.plot(time, contrast_values, label='Contrast over Time', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Contrast')
    plt.title('Contrast Time Series')
    plt.legend()

    # Plot the sharpness time series
    plt.subplot(3, 1, 3)
    plt.plot(time, sharpness_values, label='Sharpness over Time', color='red')
    plt.xlabel('Time (s)')
    plt.ylabel('Sharpness')
    plt.title('Sharpness Time Series')
    plt.legend()

    plt.tight_layout()

    os.makedirs(save_path_image+"/"+movie_name, exist_ok=True)
    plt.savefig(save_path_image+"/"+movie_name+"/"+"Brightness_contrast_sharpness.png")
    print(f"Histogram image saved in general '{save_path_image}'.")


""" 
# Example usage:
movie_name = 'After_The_Rain'

movie_names = ['After_The_Rain_exp', 'Between_Viewings_exp', 'Big_Buck_Bunny_exp', 'Chatter_exp', 
               'Damaged_Kung_Fu_exp', 'First_Bite_exp', 'Lesson_Learned_exp', 'Payload_exp', 
               'Riding_The_Rails_exp', 'Sintel_exp', 'Spaceman_exp','Superhero_exp', 'Tears_of_Steel_exp', 
               'The_secret_number_exp', 'To_Claire_From_Sonny_exp', 'You_Again_exp']

video_path = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/FilmFiles/After_The_Rain_exp.mp4'
save_path_dataframe = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX'
save_path_image = '/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Plots'
analyze_video_characteristics(video_path, save_path_dataframe, save_path_image, movie_name)
"""
