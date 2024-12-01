import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Step 1: Open the video
video_path = '/Users/shrinidhivelan/Documents/Projet CSE I/data/FilmFiles/Sintel_exp.mp4'  # Replace with your video file
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# Step 2: Initialize variables to store HOG features
hog_features = []
hog_images = []

# Step 3: Process each frame and compute HOG features
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Compute HOG features
    fd, hog_image = hog(gray_frame, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

    # Rescale HOG image for better visualization
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    # Store the features and the HOG image
    hog_features.append(fd)
    hog_images.append(hog_image_rescaled)

# Step 4: Save HOG feature vectors and images (if needed)
# Convert the list of HOG features into a numpy array
hog_features = np.array(hog_features)

# Save the HOG feature vectors to a CSV file
import pandas as pd
hog_df = pd.DataFrame(hog_features)
hog_df.to_csv("hog_feature_data.csv", index=False)

print("HOG feature data saved as 'hog_feature_data.csv'.")

# Step 5: Show a few HOG images for visualization
for i, hog_img in enumerate(hog_images[:5]):  # Show the first 5 HOG images
    plt.figure(figsize=(6, 6))
    plt.imshow(hog_img, cmap='gray')
    plt.title(f'HOG Image {i + 1}')
    plt.axis('off')
    plt.show()

# Step 6: Release the video capture object
cap.release()
