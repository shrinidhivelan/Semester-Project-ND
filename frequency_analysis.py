import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
import pandas as pd
import os


def analyze_brightness_frequency(video_path, save_path_dataframe, save_path_image, movie_name='Movie', window_duration=1, step_fraction=0.5):
    """
    Analyze the brightness time series of a video and compute the frequency spectrum using a sliding window approach.

    Args:
        video_path (str): Path to the input video file.
        save_path (str): Directory to save the frequency time series data.
        window_duration (float): Duration of each sliding window in seconds (default is 1 second).
        step_fraction (float): Fraction of the window duration for step size (default is 0.5 for 50% overlap).
    """
    # Step 1: Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    # Step 2: Initialize variables
    brightness_values = []
    frame_count = 0

    # Get FPS and check validity
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30  # Default FPS

    # Calculate window and step sizes in frames
    window_size = int(fps * window_duration)
    step_size = int(window_size * step_fraction)

    # Step 3: Process video frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate average brightness using the V channel
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        avg_brightness = np.mean(hsv_frame[:, :, 2])  # V (brightness) channel
        brightness_values.append(avg_brightness)

        frame_count += 1

    cap.release()

    # Step 4: Prepare brightness time series
    brightness_values = np.array(brightness_values)
    time_series = np.arange(frame_count) / fps  # Convert frame indices to time in seconds

    # Step 5: Perform sliding window Fourier Transform
    freq_time_series = []
    for start_frame in range(0, frame_count - window_size, step_size):
        # Extract brightness values for the window
        window_brightness = brightness_values[start_frame:start_frame + window_size]

        # Compute Fourier Transform
        N = len(window_brightness)
        fft_values = fft(window_brightness)
        frequencies = fftfreq(N, d=1 / fps)

        # Extract positive frequencies and magnitudes
        positive_freqs = frequencies[:N // 2]
        positive_magnitudes = np.abs(fft_values[:N // 2])

        # Store the frequency time series
        for f, m in zip(positive_freqs, positive_magnitudes):
            freq_time_series.append([start_frame / fps, f, m])  # Time, Frequency, Magnitude

    # Step 6: Save frequency time series data
    freq_df = pd.DataFrame(freq_time_series, columns=["Time (s)", "Frequency (Hz)", "Magnitude"])
    os.makedirs(save_path_dataframe+"/"+movie_name, exist_ok=True)
    csv_path = os.path.join(save_path_dataframe+"/"+movie_name, "frequency_time_series_data.csv")
    freq_df.to_csv(csv_path, index=False)
    print(f"Frequency time series data saved as '{csv_path}'.")

    # Step 7: Plot the frequency time series
    plt.figure(figsize=(12, 6))
    scatter = plt.scatter(
        freq_df["Time (s)"], freq_df["Magnitude"], 
        c=freq_df["Frequency (Hz)"], cmap='jet', s=10
    )
    plt.title("Frequency Time Series")
    plt.xlabel("Time (s)")
    plt.ylabel("Magnitude")
    plt.colorbar(scatter, label="Frequency (Hz)")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    os.makedirs(save_path_image+"/"+movie_name, exist_ok=True)
    plot_path = os.path.join(save_path_image+"/"+movie_name, "frequency_time_series_plot.png")

    plt.savefig(plot_path)
    print(f"Frequency time series plot saved as '{plot_path}'.")


""" 
# Example Usage
movie_name = 'Sintel_exp'
video_file_path = "/Users/shrinidhivelan/Documents/Projet CSE I/data/FilmFiles/Sintel_exp.mp4"
output_directory = "/Users/shrinidhivelan/Documents/Projet CSE I/data/Output"
analyze_brightness_frequency(video_file_path, output_directory, movie_name)
"""
