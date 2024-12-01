import cv2 as cv
from ultralytics import YOLO
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os


def analyze_object_frequencies(video_path, save_path_dataframe, save_path_image, model_path, movie_name='Sintel_exp'):
    """
    Analyze object frequencies in a video using YOLO for object detection.
    Save the detection results and overall object frequency to CSV files, and plot the object frequency over time.

    Args:
        video_path (str): Path to the input MP4 video file.
        save_path_dataframe (str): Directory path to save the output CSV files.
        save_path_image (str): Directory path to save the frequency histogram image.
        model_path (str): Path to the YOLO model file.
        movie_name (str): Name of the video/movie being analyzed (used for file naming).
    """
    # Step 1: Load YOLO model and open video
    model = YOLO(model_path)
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    # Step 2: Initialize variables
    frame_data = []
    overall_frequency = Counter()
    frame_count = 0

    # Get FPS and validate it
    fps = cap.get(cv.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Perform object detection
        results = model.predict(frame, stream=False)
        id_counts = count_objects_per_frame(results)

        # Update the overall frequency counter
        overall_frequency.update(id_counts)

        # Append frame-specific data
        frame_data.append({
            "Frame Index": frame_count,
            "Time (s)": frame_count / fps,
            "Detected Objects": dict(id_counts),
        })

    cap.release()

    # Step 3: Prepare data for saving
    df = pd.DataFrame(frame_data)

    # Overall frequency table
    overall_freq_table = {
        "Object Class": list(overall_frequency.keys()),
        "Frequency": list(overall_frequency.values())
    }
    overall_freq_df = pd.DataFrame(overall_freq_table)

    # Ensure save directories exist
    os.makedirs(f"{save_path_dataframe}/{movie_name}", exist_ok=True)
    os.makedirs(f"{save_path_image}/{movie_name}", exist_ok=True)

    # Save DataFrame to CSV
    df.to_csv(f"{save_path_dataframe}/{movie_name}/object_detection.csv", index=False)
    overall_freq_df.to_csv(f"{save_path_dataframe}/{movie_name}/overall_object_frequencies.csv", index=False)

    print(f"Data saved as '{save_path_dataframe}/{movie_name}/object_detection.csv' and '{save_path_dataframe}/{movie_name}/overall_frequencies.csv'.")

    # Step 4: Plot the overall object frequency
    plt.figure(figsize=(10, 6))
    plt.bar(overall_frequency.keys(), overall_frequency.values(), color='skyblue')
    plt.xlabel("Object Classes")
    plt.ylabel("Frequency")
    plt.title("Overall Object Frequency")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save plot
    plt.savefig(f"{save_path_image}/{movie_name}/object_frequency.png")
    plt.show()

    print(f"Frequency histogram saved at '{save_path_image}/{movie_name}/object_frequency.png'.")


def count_objects_per_frame(results):
    """
    Count the occurrences of each object class in YOLO detection results for a single frame.

    Args:
        results: YOLO detection results for the current frame.

    Returns:
        dict: A dictionary with object class names as keys and their counts as values.
    """
    detection_classes = results[0].names
    id_counts = {}
    for result in results:
        for data in result.boxes.data.tolist():
            obj_id = int(data[5])  # Object class ID
            class_name = detection_classes[obj_id]
            if class_name in id_counts:
                id_counts[class_name] += 1
            else:
                id_counts[class_name] = 1
    return id_counts

""" 
# Example Usage
video_path = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/FilmFiles/Sintel_exp.mp4"
save_path_dataframe = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/DataframeX"
save_path_image = "/Volumes/LaCie/EPFL/Mastersem3/SemesterProjectND/Plots"
model_path = "./yolov8n.pt"
movie_name = "Sintel_exp"

analyze_object_frequencies(video_path, save_path_dataframe, save_path_image, model_path, movie_name)

"""