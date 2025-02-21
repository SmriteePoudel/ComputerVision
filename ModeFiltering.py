from google.colab import drive
drive.mount('/content/drive')

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import median_filter # Import the median_filter function

file_path = '/content/drive/MyDrive/CV lab/image.jpeg'
image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Could not load image. Please check the file path and ensure the image file is not corrupted.")
else:
    median_filtered = cv2.medianBlur(image, 5)

    # Apply a mode filter using a 5x5 window
    mode_filtered = median_filter(image, size=5) # Calculate the mode filtered image

    # Plotting the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.title('Mode Filtered Image')
    plt.imshow(mode_filtered, cmap='gray') # Use the calculated mode_filtered variable
    plt.show()
