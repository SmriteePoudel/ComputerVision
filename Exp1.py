from google.colab import drive
drive.mount('contentdrive')
import cv2
from matplotlib import pyplot as plt
# Function to display images
def display_image(title, image, cmap=None)
    plt.imshow(image, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()
# a. To read an image
file_path = 'contentdriveMyDriveCV labimage.jpeg'
image = cv2.imread(file_path)
# b. To show an image
display_image('Original Image', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# c. Convert RGB to Gray Scale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# d. Read RGB values of a pixel
x, y = 50, 50
(b, g, r) = image[y, x]
print(f'RGB values at ({x}, {y}): (R: {r}, G: {g}, B: {b})')

display_image('Gray Scale Image', gray_image, cmap='gray')
# e. Convert Gray Scale to Binary
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
display_image('Binary Image', binary_image, cmap='gray')
# f. Perform Image Crop
cropped_image = image[500:900, 400:700]
display_image('Cropped Image', cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# g. Perform Image Resize
resized_image = cv2.resize(image, (200, 300))
display_image('Resized Image', cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
# h. Rotation of an image
def rotate_image(image, angle):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
# Angles to rotate
angles = [180, 90, 270, 360, 45]
titles = ['180 deg', '90 deg', '270 deg', '360 deg', '45 deg']

# Create subplots
fig, axs = plt.subplots(1, len(angles), figsize=(15, 5))

for i, angle in enumerate(angles):
    rotated_image = rotate_image(image, angle)
    axs[i].imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))
    axs[i].set_title(f'Rotated {titles[i]}')
    axs[i].axis('off')
plt.show()

# i. Histogram Equalization
equalized_image = cv2.equalizeHist(gray_image)
display_image('Histogram Equalized Image', equalized_image, cmap='gray')


# Plotting the original and equalized histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(gray_image.ravel(), 256, [0, 256])
plt.title('Original Histogram')

plt.subplot(1, 2, 2)
plt.hist(equalized_image.ravel(), 256, [0, 256])
plt.title('Equalized Histogram')

