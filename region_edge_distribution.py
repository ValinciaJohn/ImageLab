import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def kmeans_segmentation(image, K=2):
    pixel_values = image.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(pixel_values)
    labels = kmeans.predict(pixel_values)
    centers = np.uint8(kmeans.cluster_centers_)

    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image, labels.reshape(image.shape[:2])

def sobel_edge_detection(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_combined = np.sqrt(sobel_x**2 + sobel_y**2)
    sobel_combined = np.uint8(sobel_combined)
    return sobel_combined

K = 2  
segmented_image, labels = kmeans_segmentation(image, K)
edges = sobel_edge_detection(image)

unique, counts = np.unique(labels, return_counts=True)

# Plot region size distribution 
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.bar(range(len(unique)), counts)
plt.title("Region Size Distribution for K-means")
plt.xlabel("Cluster Index")
plt.ylabel("Size (Number of Pixels)")
plt.xticks(range(len(unique)))

# Plot edge density 
plt.subplot(1, 2, 2)
plt.hist(edges.ravel(), bins=256, color='gray', alpha=0.7)
plt.title("Edge Density Distribution from Sobel")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
