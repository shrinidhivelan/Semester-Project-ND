import cv2
import numpy as np
import pandas as pd
import os

def extract_and_detect_zoom(video_path, csv_save_path, movie_name):
    """
    Extracts frame data and detects zooming in a video by comparing histograms of consecutive frames.
    The results are saved to a CSV file with frame index, time (in seconds), and zoom status (Yes/No).
    
    Parameters:
    - video_path (str): Path to the input video file.
    - csv_save_path (str): Path to save the zoom detection results as a CSV file.
    - movie_name (str): The name of the movie to create specific folders for saving the CSV file.
    """
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Initialize data storage for results
    zoom_results = {
        'Frame Index': [],
        'Time (s)': [],
        'Zoom': []
    }
    frame_count = 0

    # Get FPS and check validity
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30  # Default FPS if the value is invalid

    # Read the first frame and convert to grayscale
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Unable to read the first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Define a function to compute the histogram of a frame
    def compute_histogram(frame):
        # Convert the frame to grayscale (or use specific color channels)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Compute histogram using 256 bins for grayscale intensity
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        # Normalize the histogram to compare between frames
        hist = hist / hist.sum()
        return hist

    # Compute histogram for the first frame
    prev_hist = compute_histogram(prev_frame)

    # Extract frames and compute histograms
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Update frame count and time (time in seconds)
        frame_count += 1
        frame_time = frame_count / fps  # Assuming video runs at 30 fps

        # Compute the histogram for the current frame
        curr_hist = compute_histogram(frame)

        # Calculate the correlation (or other distance metrics) between the histograms
        hist_diff = cv2.compareHist(prev_hist, curr_hist, cv2.HISTCMP_CORREL)

        # A high correlation (close to 1) indicates little change between frames
        # A low correlation indicates a significant change (possible zoom-in or zoom-out)
        zoom_detected = False
        if hist_diff < 0.8:  # Threshold based on your experiment
            zoom_detected = True

        # Store the result
        zoom_results['Frame Index'].append(frame_count)
        zoom_results['Time (s)'].append(frame_time)
        zoom_results['Zoom'].append(zoom_detected)

        # Update previous histogram for the next comparison
        prev_hist = curr_hist

    cap.release()

    # Convert the results to a DataFrame
    df = pd.DataFrame(zoom_results)

    # Ensure the save path exists, create it if it doesn't
    movie_path = os.path.join(csv_save_path, movie_name)
    if not os.path.exists(movie_path):
        os.makedirs(movie_path)

    # Save the results as a CSV file
    csv_file = os.path.join(movie_path, "ZoomDetectionResults.csv")
    df.to_csv(csv_file, index=False)
    print(f"Zoom detection results saved as '{csv_file}'.")

