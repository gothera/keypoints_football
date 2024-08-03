import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

from PIL import Image

def get_image_paths(dataset):
    directory = f"/Users/cosmincojocaru/keypoints_football/keypoints_football/keypoints_dataset/keypoints_dataset/{dataset}"
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp')
    image_names = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(image_extensions):
                image_names.append(file.removesuffix('.jpg'))
    
    return image_names

def plot_image_with_keypoints(dataset, img_id):
    # Read the image
    image_path = f'./keypoints_dataset/keypoints_dataset/{dataset}/{img_id}.jpg'
    keypoints_file_path = f'./keypoints_dataset/keypoints_dataset/{dataset}/{img_id}.txt'

    img = Image.open(image_path)

    # If keypoints_file_path is not provided, assume it's the same as image_path but with .txt extension
    if keypoints_file_path is None:
        keypoints_file_path = os.path.splitext(image_path)[0] + '.txt'

    # Read keypoints from the file
    try:
        with open(keypoints_file_path, 'r') as f:
            keypoints_data = f.read().strip().split()
    except UnicodeDecodeError:
        print(f"Error: {keypoints_file_path} is not a valid text file. Please provide the correct path to the keypoints file.")
        return
    except FileNotFoundError:
        print(f"Error: Keypoints file not found at {keypoints_file_path}")
        return

    # Extract keypoints (YOLO v8 pose format)
    keypoints_data = [float(x) for x in keypoints_data[1:]]  # Skip class index
    bbox = np.array(keypoints_data[:4])
    keypoints = np.array(keypoints_data[4:]).reshape(-1, 3)
    
    non_origin_mask = np.any(keypoints != 0.0, axis=1)
    count = np.sum(non_origin_mask)
    print(f'Found {count} visible points out of {len(keypoints)}')

    # Get image dimensions
    width, height = img.size

    # Scale bbox and keypoints to image dimensions
    bbox[0:2] = bbox[0:2] - bbox[2:4] / 2  # xy top-left corner
    bbox[2:4] = bbox[0:2] + bbox[2:4]  # xy bottom-right corner
    bbox = bbox * np.array([width, height, width, height])
    keypoints[:, 0] *= width
    keypoints[:, 1] *= height

    # Create a figure with the same size as the image
    dpi = 100
    figsize = (img.width / dpi, img.height / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display the image
    ax.imshow(img)
    
    # Extract x and y coordinates from keypoints
    x = [point[0] for point in keypoints]
    y = [point[1] for point in keypoints]

    for idx, p in enumerate(zip(x, y)):
        ax.text(p[0], p[1], idx,
            color='white', fontweight='bold', fontsize=12, ha='center', va='center')    
    # Scatter the keypoints
    ax.scatter(x, y, c='red', s=10)
    
    # Remove axes
    ax.axis('off')
    
    # Ensure the figure fills the entire area without padding
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # Save the figure
    # plt.savefig(f'./keypoints_dataset/keypoints_dataset/{dataset}/{img_id}_annotated.jpg',  dpi=dpi, bbox_inches='tight', pad_inches=0)
    # plt.close()

    plt.imshow(img)
    plt.show()


if __name__ == "__main__":
    image_names = get_image_paths("train")
    for img_name in image_names:
        print(img_name)
        plot_image_with_keypoints('train', img_name)
