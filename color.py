import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def extract_and_save_color_histograms(video_path, csv_save_path, image_save_path, movie_name):
    """
    Extracts color histograms from a video and saves the histogram data to a CSV file
    and the histogram plots of the first frame to an image file.

    Parameters:
    - video_path (str): Path to the input video file.
    - csv_save_path (str): Path to save the extracted histogram data as a CSV file.
    - image_save_path (str): Path to save the histogram plots of the first frame as an image.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Initialize data storage for histograms
    histogram_values = {
        'Frame Index': [],
        'Time (s)': [],
        'Histogram B': [],
        'Histogram G': [],
        'Histogram R': []
    }
    frame_count = 0

    # Get FPS and check validity
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30  # Default FPS if the value is invalid

    # Extract histograms for each frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Calculate histograms for BGR channels
        hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
        hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])

        # Store aggregated histogram values
        histogram_values['Frame Index'].append(frame_count)
        histogram_values['Time (s)'].append(frame_count / fps)
        histogram_values['Histogram B'].append(np.sum(hist_b))
        histogram_values['Histogram G'].append(np.sum(hist_g))
        histogram_values['Histogram R'].append(np.sum(hist_r))

        frame_count += 1

    cap.release()

    # Convert the data to a DataFrame
    df = pd.DataFrame(histogram_values)
    csv_save_path = csv_save_path+"/"+movie_name+"/"+"ColorDataframe.csv"
    df.to_csv(csv_save_path, index=False)
    print(f"Histogram data saved as '{csv_save_path}'.")

    # Read the first frame for plotting histograms
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()

    if not ret:
        print("Error: Unable to read the first frame from the video.")
        return

    # Calculate histograms for the first frame
    hist_b = cv2.calcHist([first_frame], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([first_frame], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([first_frame], [2], None, [256], [0, 256])

    # Plot histograms
    plt.figure(figsize=(12, 6))

    # Blue histogram
    plt.subplot(3, 1, 1)
    plt.plot(hist_b, color='blue')
    plt.title("Blue Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Green histogram
    plt.subplot(3, 1, 2)
    plt.plot(hist_g, color='green')
    plt.title("Green Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    # Red histogram
    plt.subplot(3, 1, 3)
    plt.plot(hist_r, color='red')
    plt.title("Red Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.savefig(image_save_path+"/"+movie_name+"/"+"ColorHistogram.png")
    print(f"Histogram image saved as '{image_save_path}'.")

# Example usage:
# extract_and_save_color_histograms(
#     video_path='/Users/shrinidhivelan/Documents/Projet CSE I/data/FilmFiles/Sintel_exp.mp4',
#     csv_save_path='/path/to/save_data.csv',
#     image_save_path='/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Plots', 
#     movie_name = ''
#     
# )
