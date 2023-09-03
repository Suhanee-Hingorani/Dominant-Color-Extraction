# Import necessary libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

# Read an image named "flower_field.jpg" using OpenCV
im = cv2.imread("flower_field.jpg")

# Convert the image from BGR to RGB color format
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

# Get the dimensions of the image
r, c = im.shape[:2]

# Define the desired output height
out_r = 200

# Resize the image to have the desired output height while maintaining aspect ratio
im = cv2.resize(im, (int(out_r * float(c) / r), out_r))

# Reshape the image into a 2D array of pixels
pixels = im.reshape((-1, 3))

# Print the shape of the pixel array
print(pixels.shape)

# Display the resized image using Matplotlib
plt.imshow(im)
plt.show()

# Initialize a KMeans clustering model with 8 clusters
km = KMeans(n_clusters=8)

# Fit the KMeans model to the pixel data
km.fit(pixels)

# Get the cluster centers (colors) as an array of integers
colors = np.asarray(km.cluster_centers_, dtype='uint8')

# Print the colors representing the cluster centers
print(colors)

# Calculate the percentage of pixels assigned to each cluster
per = np.asarray(np.unique(km.labels_, return_counts=True)[1], dtype='float32')
per = per / pixels.shape[0]

# Print the percentage of pixels in each cluster
print(per)

# Create a figure to display the dominant colors
plt.figure(0)

# Loop through the dominant colors and display them as patches
for ix in range(colors.shape[0]):
    patch = np.ones((100, 100, 3), dtype=np.uint8) * colors[ix]
    plt.subplot(1, colors.shape[0], ix + 1)
    plt.axis("off")
    plt.imshow(patch)

# Show the dominant colors
plt.show()

# Find the index of the dominant cluster (the one with the highest percentage)
dominant_cluster_index = np.argmax(per)

# Get the color of the dominant cluster
dominant_color = colors[dominant_cluster_index]

# Create a patch with the dominant color
dominant_patch = np.ones((100, 100, 3), dtype=np.uint8) * dominant_color

# Display the dominant color as a patch
plt.figure()
plt.imshow(dominant_patch)
plt.axis("off")
plt.show()
