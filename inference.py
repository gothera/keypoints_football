import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
import matplotlib.pyplot as plt

def plot_keypoints(image, keypoints, radius=3, color=(0, 0, 255)):
    """Plot keypoints on the image."""
    for idx, (x, y), in enumerate(keypoints):
        cv2.circle(image, (int(x), int(y)), 5,
                 color,)
        cv2.putText(image, str(idx), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    return image
    

# Load the YOLOv8n-pose model
model = YOLO('last.pt')

# Open the input video
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create VideoWriter object
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

no = 0
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()
    if not success:
        break
    # frame = cv2.resize(frame, (640, 360))
    no += 1
    if no < 100:
        continue
    results = model(frame, conf=0.25)
    # Get keypoints
    keypoints = results[0].keypoints.xy[0].cpu().numpy()
    # Plot keypoints
    image = plot_keypoints(frame, keypoints, color=(255, 0, 0))

    # Write the annotated frame
    out.write(frame)
    # Write the annotated frame
    # plt.imshow(frame)
    # plt.show()

# Release video capture and writer objects
cap.release()
out.release()
print(f"Annotated video saved to {output_path}")