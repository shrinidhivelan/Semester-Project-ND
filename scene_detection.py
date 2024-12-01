import torch
from torch.autograd import Variable as V
import torchvision.models as models
from torchvision import transforms as trn
from torch.nn import functional as F
from PIL import Image

import cv2
import os
import pandas as pd
import numpy as np
from collections import deque

def analyze_scene_classification(video_path, save_path_dataframe, movie_name='Movie', window_duration=1, step_fraction=0.5, arch='resnet50'):
    """
    Analyze scene classification time series of a video using a sliding window approach.

    Args:
        video_path (str): Path to the input video file.
        save_path_dataframe (str): Directory to save the scene classification time series data.
        movie_name (str): Name of the movie (used for saving results).
        window_duration (float): Duration of each sliding window in seconds.
        step_fraction (float): Fraction of the window duration for step size (default is 0.5 for 50% overlap).
        arch (str): CNN architecture to use (default is 'resnet50').
    """
    # Load the pre-trained model
    model_file = '/Users/shrinidhivelan/Desktop/Object Detection/Scene classification/resnet50_places365.pth.tar' #f'Scene classification/{arch}_places365.pth.tar'
    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()

    # Load the class labels
                #'Scene classification/categories_places365.txt'
    file_name = '/Users/shrinidhivelan/Desktop/Object Detection/Scene classification/categories_places365.txt'
    classes = []
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    classes = tuple(classes)

    # Define image transformer
    centre_crop = trn.Compose([
        trn.Resize((256, 256)),
        trn.CenterCrop(224),
        trn.ToTensor(),
        trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Step 1: Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Unable to open video at {video_path}")
        return

    # Step 2: Initialize variables
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print("Error: Unable to retrieve FPS. Setting default FPS to 30.")
        fps = 30

    # Calculate window and step sizes in frames
    window_size = int(fps * window_duration)
    step_size = int(window_size * step_fraction)

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    scene_probabilities = []
    sliding_window = deque(maxlen=window_size)

    # Step 3: Process video frames
    for frame_idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL image and preprocess
        img = V(centre_crop(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))).unsqueeze(0))

        # Get model predictions
        with torch.no_grad():
            logit = model.forward(img)
            probs = F.softmax(logit, 1).data.squeeze().cpu().numpy()
        
        sliding_window.append(probs)
        if len(sliding_window) == window_size:
            # Calculate average probabilities for the current window
            avg_probs = np.mean(np.array(sliding_window), axis=0)
            top_classes = np.argsort(avg_probs)[::-1][:5]
            scene_probabilities.append({
                "Time (s)": frame_idx / fps,
                "Top Class": classes[top_classes[0]],
                "Probability": avg_probs[top_classes[0]],
                "Top 5 Classes": [classes[i] for i in top_classes],
                "Top 5 Probabilities": avg_probs[top_classes].tolist()
            })

    cap.release()

    # Step 4: Save results to a DataFrame
    os.makedirs(save_path_dataframe, exist_ok=True)
    output_path = os.path.join(save_path_dataframe+"/"+movie_name, "Scene_time_series.csv")
    df = pd.DataFrame(scene_probabilities)
    df.to_csv(output_path, index=False)
    print(f"Scene classification time series saved to '{output_path}'.")


""" 
# Example usage
video_path = "/path/to/your/video.mp4"
save_path = "/path/to/save/results"
analyze_scene_classification(video_path, save_path, movie_name="Example_Movie")
"""
