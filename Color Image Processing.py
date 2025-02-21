import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('glassPic.jpeg')

# Convert the image to the HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the color ranges for segmentation

# Red color range (two ranges needed to cover the red hue wrap-around in HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Blue color range
lower_blue = np.array([100, 150, 70])
upper_blue = np.array([140, 255, 255])

# Green color range
lower_green = np.array([40, 70, 70])
upper_green = np.array([80, 255, 255])

# Create masks for the color ranges
mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
mask_red = cv2.add(mask_red1, mask_red2)  # Combine both masks for red

mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

# Extract the colored objects from the image
red_objects = cv2.bitwise_and(image, image, mask=mask_red)
blue_objects = cv2.bitwise_and(image, image, mask=mask_blue)
green_objects = cv2.bitwise_and(image, image, mask=mask_green)

# Convert BGR to RGB for displaying with matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
red_objects_rgb = cv2.cvtColor(red_objects, cv2.COLOR_BGR2RGB)
blue_objects_rgb = cv2.cvtColor(blue_objects, cv2.COLOR_BGR2RGB)
green_objects_rgb = cv2.cvtColor(green_objects, cv2.COLOR_BGR2RGB)

# Display the original and segmented images using matplotlib
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(red_objects_rgb)
plt.title('Red Objects')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(blue_objects_rgb)
plt.title('Blue Objects')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(green_objects_rgb)
plt.title('Green Objects')
plt.axis('off')

plt.tight_layout()
plt.show()
